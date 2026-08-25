import random

from numpy import mean
import torch
from torch import nn
from AI.Agent.turnNetwork import TurnNetwork
from AI.Agent.observation import Observation
from AI.Agent.selectionNetwork import SelectionNetwork
from AI.Agent.replayMemoryAN import ReplayMemoryAN
from AI.Agent.replayMemoryPM import ReplayMemoryPM
from constants import ABILITIES, BATCH_SIZE, COPY_DQN, DISCOUNT_FACTOR_SEL, DISCOUNT_FACTOR_TURN, EPSILON_SEL_DECAY, EPSILON_SEL_MIN, EPSILON_SELECTION, EPSILON_TURN, EPSILON_TURN_DECAY, EPSILON_TURN_MIN, SELECTION_LEARNING_RATE, SELECTION_REPLAY_DATA, TURN_LEARNING_RATE, TURN_REPLAY_DATA, WARRIOR_QUANTITY

class PlayerAI:
    def __init__(self):
        self.name = "DqnPlayer"
        self.epsilon_sel = EPSILON_SELECTION
        self.epsilon_turn = EPSILON_TURN
        self.selection_network = SelectionNetwork()
        self.target_selection_network = SelectionNetwork()
        self.target_selection_network.load_state_dict(self.selection_network.state_dict())
        self.optimizer = torch.optim.Adam(self.selection_network.parameters(),lr = SELECTION_LEARNING_RATE)
        self.replay_memory_sel = ReplayMemoryPM(SELECTION_REPLAY_DATA)
        self.turn_network = TurnNetwork()
        self.target_turn_network = TurnNetwork()
        self.target_turn_network.load_state_dict(self.turn_network .state_dict())
        self.optimizer2 = torch.optim.Adam(self.turn_network.parameters(),lr = TURN_LEARNING_RATE)
        self.replay_memory_turn = ReplayMemoryAN(TURN_REPLAY_DATA)
        self.replayed_selection = 0
        self.replayed_turn = 0
        
    def remember_selection(self,c_state,action,reward,next_c_state,done):
        self.replay_memory_sel.push(c_state,action,reward,next_c_state,done)
        
    def remember_turn(self,observation,action,reward,next_observation,done):
        self.replay_memory_turn.push(observation,action,reward,next_observation,done)
        
    def replay_selection(self):
        if len(self.replay_memory_sel) < BATCH_SIZE:
            return None
        self.replayed_selection += 1
        batch,indices,weights = self.replay_memory_sel.sample(BATCH_SIZE)
        weights = torch.tensor(weights,dtype=torch.float32)
        states = [
            x.state.encode_choose_state()
            for x in batch
        ]
        actions_b = [x.action for x in batch]
        rewards = [x.reward for x in batch]
        next_states = self._encode_next_states(batch,lambda s: s.encode_choose_state(),
                                 self.selection_network.network[0].in_features)
        dones = [x.done for x in batch]
        states = torch.tensor(states, dtype=torch.float32)
        actions_b = torch.tensor(actions_b, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        qvalues = self.selection_network(states)
        q_selected = qvalues.gather(1,actions_b.unsqueeze(1)).squeeze(1)
        td_errors = []
        with torch.no_grad():
            next_actions = self.selection_network(next_states).argmax(
                dim=1
            )
            next_qvalues = self.target_selection_network(next_states).gather(1,next_actions.unsqueeze(1)).squeeze(1)
            target = (
                rewards
                + DISCOUNT_FACTOR_SEL
                * next_qvalues
                * (~dones)
            )
            for q_select,tar in zip (q_selected,target):
                td_errors.append(abs(q_select-tar))
        loss = self.loss_function(q_selected,target,weights)
        self._optimize_step(loss, self.optimizer, self.selection_network, 
                            self.target_selection_network, "replayed_selection")
        self.replay_memory_sel.update_priorities(indices, td_errors)
        return loss.item()
    
    def replay_turn(self):
            if len(self.replay_memory_turn) < BATCH_SIZE:
                return None
            self.replayed_turn += 1
            #Recoges las experencias que usaremos como base
            batch,indices,weights = self.replay_memory_turn.sample(BATCH_SIZE)
            weights = torch.tensor(weights,dtype=torch.float32)
            #Divides los elementos de cada experiencia
            states = [x.state.normalize() for x in batch]
            warrior_mask = torch.tensor([
                [warrior is not None for warrior in x.state.pl_warriors] for x in batch], dtype=torch.bool)
            actions_b = [x.action for x in batch]
            next_states = self._encode_next_states(batch,lambda s: s.normalize(),
                                                   self.turn_network.network[0].in_features)
            rewards = [x.reward for x in batch]
            dones = [x.done for x in batch]
            #Lo conviertes a tensor
            states = torch.tensor(states,dtype=torch.float32)
            abilites_opt = ABILITIES
            actions_b = [[abilites_opt.index(action) if action is not None else 0 for action in x.action]for x in batch]
            rewards = torch.tensor(rewards,dtype = torch.float32)
            next_states = torch.tensor(next_states,dtype=torch.float32)
            dones = torch.tensor(dones,dtype=torch.bool)
            #Recogemos los qvalues de los states
            qvalues = self.turn_network(states)
            #Recogemos los qvalues de las acciones tomadas
            offsets = torch.tensor([0, 6, 12])
            actions_b = torch.tensor(actions_b, dtype=torch.long)
            actions_global = actions_b + offsets
            q_selected = qvalues.gather(1, actions_global)
            q_selected = q_selected * warrior_mask.float()
            q_selected = q_selected.sum(dim=1)
            target = self._multi_agent_double_dqn_target(batch, next_states, rewards, dones)
            td_errors = []
            with torch.no_grad():
                for q_select,tar in zip(q_selected,target):
                    td_errors.append(abs(q_select-tar))
            #Caculamos la perdida por haber tomado la decisión que tomamos
            loss = self.loss_function(q_selected,target,weights)
            self._optimize_step(loss, self.optimizer2, self.turn_network, self.target_turn_network, "replayed_turn")
            self.replay_memory_turn.update_priorities(indices, td_errors)
            return loss.item()
        
    def loss_function(self,input,target,weigths):
        loss = nn.SmoothL1Loss(reduction="none") 
        output = loss(input,target)
        output = (output*weigths).mean()
        return output 
     
    def selection(
        self,
        cs
    ):
        choose_state = cs.encode_choose_state()
        choose_state = torch.tensor(
            choose_state,
            dtype=torch.float32
        )
        logits = self.selection_network(
            choose_state.unsqueeze(0)
        ).squeeze(0)
        masked_logits = self.mask_selection(
            logits,
            cs.pl_disposition
        )
        valid_actions = self._valid_indices(masked_logits)
        if random.random() < self.epsilon_sel or cs.opp_initial_warrior == 0:
            action = random.choice(valid_actions.tolist())
        else:
            action = torch.argmax(masked_logits).item()
        warrior_index = action // 3
        position = action % 3
        return cs.pl_warriors[warrior_index + 1].id, position,action
    
    def turn(self, observation):
        observation_enc = observation.normalize()
        observation_enc = torch.tensor(
            observation_enc,
            dtype=torch.float32
        )
        logits = self.turn_network(observation_enc.unsqueeze(0)).squeeze(0)
        masked_logits = self.mask_turn(observation.pl_warriors,observation.opp_disposition,logits)
        abilities_opt = ABILITIES
        actions = []
        for pos, warrior in enumerate(observation.pl_warriors):
            # Guerrero muerto / inexistente
            if warrior is None:
                actions.append(None)
                continue
            start = pos * 6
            end = start + 6
            valid_actions = self._valid_indices(masked_logits[start:end])
            if len(valid_actions) == 0:
                actions.append(None)
                continue
            if random.random() < self.epsilon_turn:
                action = random.choice(valid_actions.tolist())
            else:
                action = torch.argmax(masked_logits[start:end]).item()
            actions.append(abilities_opt[action])
        return actions
    
    def mask_turn(self, warriors,enemy_disp, logits):
        mask = torch.ones(6 * 3,dtype=torch.bool)
        for pos, warrior in enumerate(warriors):
            # Si el guerrero está muerto/inexistente,
            if warrior == None:
                for ability_index in range(6):
                    mask[pos * 6 + ability_index] = False
                continue
            else:
                for ability_index in range(4):
                    if(warrior.cooldown_abilities[ability_index] == True):
                        mask[pos * 6 + ability_index] = False
            for ability_pos, ability in enumerate(warrior.warrior_data.abilities):
                # Las habilidades sin objetivos no dependen de la posición enemiga
                if not ability.target_positions:
                    continue
                # La habilidad es válida si al menos uno de sus objetivos existe
                has_valid_target = any(
                    enemy_disp[target_pos] != 0
                    for target_pos in ability.target_positions
                )
                if not has_valid_target:
                    mask[pos * 6 + ability_pos] = False
            if pos == 0:
                # En posición 1 no puede moverse hacia atrás
                mask[pos * 6 + 5] = False  # movNeg
            elif pos == 2:
                # En posición 3 no puede moverse hacia delante
                mask[pos * 6 + 4] = False  # movPos

        return logits.masked_fill(
            ~mask,
            float("-inf")
        )
                
    def mask_selection(self, logits, disposition):
        mask = torch.ones(
            WARRIOR_QUANTITY * 3,
            dtype=torch.bool
        )
        for position, warrior in enumerate(disposition):
            # La posición está ocupada
            if warrior is not None:
                warrior_index = warrior.warrior_data.id - 1
                # No podemos volver a seleccionar ese guerrero
                for pos in range(3):
                    action_index = warrior_index * 3 + pos
                    mask[action_index] = False
                # Tampoco podemos colocar otro guerrero en esa posición
                for wi in range(WARRIOR_QUANTITY):
                    action_index = wi * 3 + position
                    mask[action_index] = False
        return logits.masked_fill(
            ~mask,
            float("-inf")
        ) 
    def _network_specs(self):
        return [
            (self.selection_network, self.target_selection_network,
            self.optimizer, self.replay_memory_sel,
            "epsilon_sel", "replayed_selection"),

            (self.turn_network, self.target_turn_network,
            self.optimizer2, self.replay_memory_turn,
            "epsilon_turn", "replayed_turn"),
        ]
        
    def save_model(self, path1, path2):
        for path, (net, target_net, opt, replay_memory,
                eps_attr, replayed_attr) in zip(
                    (path1, path2), self._network_specs()):

            torch.save({
                "dqn": net.state_dict(),
                "targetdqn": target_net.state_dict(),
                "optimizer": opt.state_dict(),
                "epsilon": getattr(self, eps_attr),
                "replayed": getattr(self, replayed_attr),
                "replay_memory": replay_memory.state_dict(),
            }, path)

    def load_model(self, path1, path2):
        for path, (net, target_net, opt, replay_memory,
                eps_attr, replayed_attr) in zip(
                    (path1, path2), self._network_specs()):

            checkpoint = torch.load(path,weights_only=False)

            net.load_state_dict(checkpoint["dqn"])
            target_net.load_state_dict(checkpoint["targetdqn"])
            opt.load_state_dict(checkpoint["optimizer"])

            setattr(self, eps_attr, checkpoint["epsilon"])
            setattr(self, replayed_attr, checkpoint["replayed"])

            replay_memory.load_state_dict(
                checkpoint["replay_memory"])
    
    def update_epsilon(self):
        self.epsilon_sel = max(
            EPSILON_SEL_MIN,
            self.epsilon_sel * EPSILON_SEL_DECAY
        )
        self.epsilon_turn = max(
            EPSILON_TURN_MIN,
            self.epsilon_turn * EPSILON_TURN_DECAY
        )
        
    def update_beta(self):
        self.replay_memory_sel.update_beta(self.replayed_selection)
        self.replay_memory_turn.update_beta(self.replayed_turn)
        
    @staticmethod
    def _encode_next_states(batch, encode_fn, in_features):
        return [encode_fn(x.next_state) if x.next_state is not None else [0] * in_features for x in batch]
    @staticmethod
    def _valid_indices(masked_logits):
        return torch.where(masked_logits != float("-inf"))[0]
    def _optimize_step(self, loss, optimizer, network, target_network, replayed_counter_attr):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        replayed = getattr(self, replayed_counter_attr)
        if replayed % COPY_DQN == 0:
            target_network.load_state_dict(network.state_dict())
            
    def _multi_agent_double_dqn_target(self, batch, next_states, rewards, dones):
        with torch.no_grad():
            next_masks = torch.stack([
                self.mask_turn(
                    x.next_state.pl_warriors if x.next_state is not None else [None, None, None],
                    x.next_state.opp_disposition if x.next_state is not None else [0, 0, 0],
                    torch.ones(18, dtype=torch.bool)
                )
                for x in batch]).bool()
            next_qvalues_main = self.turn_network(next_states)
            next_qvalues_main = next_qvalues_main.masked_fill(~next_masks, float("-inf"))
            next_q1 = next_qvalues_main[:, 0:6]
            next_q2 = next_qvalues_main[:, 6:12]
            next_q3 = next_qvalues_main[:, 12:18]
            next_a1 = next_q1.argmax(dim=1)
            next_a2 = next_q2.argmax(dim=1)
            next_a3 = next_q3.argmax(dim=1)
            next_actions = torch.stack([next_a1, next_a2 + 6, next_a3 + 12], dim=1)
            target_qvalues = self.target_turn_network(next_states)
            next_qvalues = target_qvalues.gather(1, next_actions)
            next_warrior_mask = torch.tensor(
                [[warrior is not None for warrior in x.next_state.pl_warriors]
                if x.next_state is not None else [False, False, False]
                for x in batch],
                dtype=torch.bool
            )
            next_qvalues = (next_qvalues * next_warrior_mask.float()).sum(dim=1)
            return rewards + DISCOUNT_FACTOR_TURN * next_qvalues * (~dones)