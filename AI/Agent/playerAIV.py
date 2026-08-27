import torch
from torch import nn
from AI.Agent.turnNetwork import TurnNetwork
from AI.Agent.selectionNetwork import SelectionNetwork
from AI.Agent.replayMemoryAN import ReplayMemoryAN
from AI.Agent.replayMemoryPM import ReplayMemoryPM
from constants import (ABILITIES, BATCH_SIZE, COPY_DQN, DISCOUNT_FACTOR, EPSILON_SEL_DECAY, EPSILON_SEL_MIN,
                        EPSILON_SELECTION, EPSILON_TURN, EPSILON_TURN_DECAY, EPSILON_TURN_MIN,
                        SELECTION_LEARNING_RATE, SELECTION_REPLAY_DATA, TURN_LEARNING_RATE,
                        TURN_REPLAY_DATA, WARRIOR_QUANTITY)


class PlayerAIV:

    def __init__(self, N, environment):
        self.N = N
        self.environment = environment
        self.name = "DqnPlayerV"
        self.epsilon_sel = EPSILON_SELECTION
        self.epsilon_turn = EPSILON_TURN
        self.selection_network = SelectionNetwork()
        self.target_selection_network = SelectionNetwork()
        self.target_selection_network.load_state_dict(self.selection_network.state_dict())
        self.optimizer = torch.optim.Adam(self.selection_network.parameters(), lr=SELECTION_LEARNING_RATE)
        self.replay_memory_sel = ReplayMemoryPM(SELECTION_REPLAY_DATA,46)
        self.turn_network = TurnNetwork()
        self.target_turn_network = TurnNetwork()
        self.target_turn_network.load_state_dict(self.turn_network.state_dict())
        self.optimizer2 = torch.optim.Adam(self.turn_network.parameters(), lr=TURN_LEARNING_RATE)
        self.replay_memory_turn = ReplayMemoryAN(TURN_REPLAY_DATA,58)
        self.replayed_selection = 0
        self.replayed_turn = 0
        
    def remember_selection_batch(self,c_state,action,reward,next_c_state,done):
        self.replay_memory_sel.push_batch(c_state,action,reward,next_c_state,done)
        
    def remember_turn_batch(self, observation, action, reward, next_observation, done,
                            alive, types, cooldowns, opp_types,
                            next_types, next_alive, next_cooldowns, next_opp_types):
        self.replay_memory_turn.push_batch(observation, action, reward, next_observation, done,
                                            alive, types, cooldowns, opp_types, 
                                            next_types, next_alive, next_cooldowns, next_opp_types)
            
    def replay_selection(self):
        if len(self.replay_memory_sel) < BATCH_SIZE: return None
        self.replayed_selection += 1
        batch, indices, weights = self.replay_memory_sel.sample(BATCH_SIZE)
        weights = torch.tensor(weights, dtype=torch.float32)

        states = batch.states
        actions = batch.actions
        rewards = batch.rewards
        next_states = batch.next_states
        dones = batch.dones

        qvalues = self.selection_network(states)
        q_selected = qvalues.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.selection_network(next_states).argmax(dim=1)
            next_qvalues = self.target_selection_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = (rewards + DISCOUNT_FACTOR * next_qvalues * (~dones))
            td_errors = torch.abs(q_selected - target)

        loss = self.loss_function(q_selected, target, weights)
        self._optimize_step(loss, self.optimizer, self.selection_network, self.target_selection_network, "replayed_selection")
        self.replay_memory_sel.update_priorities(indices, td_errors.detach().cpu().numpy())
        return loss.item()
        
    def replay_turn(self):
        if len(self.replay_memory_turn) < BATCH_SIZE:
            return None
        self.replayed_turn += 1
        batch, indices, weights = self.replay_memory_turn.sample(BATCH_SIZE)
        weights = torch.tensor(weights, dtype=torch.float32)

        states = batch.states
        warrior_mask = batch.alive
        next_states = batch.next_states
        rewards = batch.rewards
        dones = batch.dones
        actions_b = self._environment_action_to_network(batch.actions)

        current_mask_flat = self.mask_turn(                                       # NUEVO
            batch.types, batch.cooldowns, batch.alive, batch.opp_types,            # NUEVO — requiere que
            torch.ones(len(states), 18, dtype=torch.bool)                          # NUEVO   estos campos existan
        )                                                                          # NUEVO   en `batch`
        current_action_mask = (current_mask_flat != float("-inf")).view(-1, 3, 6)  # NUEVO

        qvalues = self.turn_network(states, action_mask=current_action_mask)       # CAMBIO: se pasa action_mask

        offsets = torch.tensor([0, 6, 12])
        actions_global = actions_b + offsets
        q_selected = qvalues.gather(1, actions_global)
        q_selected = q_selected * warrior_mask.float()
        q_selected = q_selected.sum(dim=1)
        target = self._multi_agent_double_dqn_target(batch, next_states, rewards, dones)
        with torch.no_grad():
            td_errors = torch.abs(q_selected - target)
        loss = self.loss_function(q_selected, target, weights)
        self._optimize_step(loss, self.optimizer2, self.turn_network, self.target_turn_network, "replayed_turn")
        self.replay_memory_turn.update_priorities(indices, td_errors.detach().cpu().numpy())
        return loss.item()
        
    def loss_function(self,input,target,weigths):
        loss = nn.SmoothL1Loss(reduction="none") 
        output = loss(input,target)
        output = (output*weigths).mean()
        return output 

    def selection(self, batch_encoded_states, disposition, opp_initial_warrior):

        states = batch_encoded_states.float()
        logits = self.selection_network(states)  # (N, WARRIOR_QUANTITY*3)
        masked_logits = self.mask_selection(logits, disposition)

        greedy = torch.argmax(masked_logits, dim=1)  # (N,)
        random_action = self._random_valid_action(masked_logits)  # (N,)

        explora = (torch.rand(self.N) < self.epsilon_sel) | (opp_initial_warrior == 0)
        action = torch.where(explora, random_action, greedy)  # (N,)

        warrior_index = action // 3
        position = action % 3
        # +1 porque en el original los ids de guerrero empiezan en 1
        return warrior_index + 1, position, action

    def mask_selection(self, logits, disposition):
        
        N = disposition.shape[0]
        mask = torch.ones(N, WARRIOR_QUANTITY * 3, dtype=torch.bool)

        ocupado = disposition > 0  # (N, 3)
        warrior_idx = (disposition - 1).clamp(min=0)  # (N, 3), evita índice -1 donde vacío

        # Invalida las 3 acciones (una por posición) del guerrero ya colocado
        accion_base = warrior_idx * 3  # (N, 3)
        for offset in range(3):
            accion = accion_base + offset
            mask.scatter_(1, accion, torch.where(ocupado, torch.zeros_like(accion, dtype=torch.bool), mask.gather(1, accion)))

        # Invalida las WARRIOR_QUANTITY acciones que colocarían algo en un slot ya ocupado
        for slot in range(3):
            slot_ocupado = ocupado[:, slot]  # (N,)
            for wi in range(WARRIOR_QUANTITY):
                accion = torch.full((N, 1), wi * 3 + slot, dtype=torch.long)
                mask.scatter_(1, accion, torch.where(slot_ocupado.unsqueeze(1),
                                                       torch.zeros_like(accion, dtype=torch.bool),
                                                       mask.gather(1, accion)))

        return logits.masked_fill(~mask, float("-inf"))

    def turn(self, batch_encoded_obs, own_disposition, own_cooldowns, own_alive, enemy_disposition):
        obs = batch_encoded_obs.float()
        logits = self.turn_network(obs)  # (N, 18)
        masked_logits = self.mask_turn(own_disposition, own_cooldowns, own_alive, enemy_disposition, logits)

        masked_3d = masked_logits.view(self.N, 3, 6)          # (N, 3, 6)
        hay_valida = (masked_3d != float("-inf")).any(dim=-1)  # (N, 3)
        greedy = torch.argmax(masked_3d, dim=-1)                # (N, 3)

        random_action = self._random_valid_action(masked_3d.reshape(self.N * 3, 6))
        random_action = random_action.view(self.N, 3)

        explora = torch.rand(self.N, 3) < self.epsilon_turn
        elegido = torch.where(explora, random_action, greedy)   # (N, 3)

        codigo = self._decode_ability_index(elegido) 
        actions = torch.where(hay_valida, codigo, torch.full_like(codigo, -1))

        return actions
    
    @staticmethod
    def _encode_next_states(batch, encode_fn, in_features):
        return [encode_fn(x.next_state) if x.next_state is not None else [0] * in_features for x in batch]
    
    @staticmethod
    def _decode_ability_index(idx_0_5):
        # índice de red: 0-3 habilidades, 4=movPos, 5=movNeg
        # código de Environment: 0-3 habilidades, 5=movPos, 6=movNeg
        return torch.where(idx_0_5 == 4, torch.full_like(idx_0_5, 5),
               torch.where(idx_0_5 == 5, torch.full_like(idx_0_5, 6), idx_0_5))

    def mask_turn(self, own_disposition, own_cooldowns, own_alive, enemy_disposition, logits):
        N = own_disposition.shape[0]
        # (N, 3) -> (N, 3, 6): un guerrero muerto invalida sus 6 acciones de golpe
        mask = own_alive.unsqueeze(-1).expand(N, 3, 6).clone()
        mask[:, :, :4] &= ~own_cooldowns
        
        table = self.environment.target_mask_por_tipo_habilidad
        target_mask_full = table[own_disposition]              # (N, 3, 4, 3)

        enemy_ocupado = (enemy_disposition > 0).unsqueeze(1).unsqueeze(1)  # (N, 1, 1, 3)
        hay_target_valido = (target_mask_full & enemy_ocupado).any(dim=-1)   # (N, 3, 4)
        sin_target = ~hay_target_valido & target_mask_full.any(dim=-1)       # (N, 3, 4)

        mask[:, :, :4] &= ~sin_target

        mask[:, 0, 5] = False   # movNeg inválido en slot 0
        mask[:, 2, 4] = False   # movPos inválido en slot 2

        mask_flat = mask.reshape(N, 18)
        return logits.masked_fill(~mask_flat, float("-inf"))

    @staticmethod
    def _random_valid_action(masked_logits):
        valid = (masked_logits != float("-inf")).float()
        valid = torch.where(valid.sum(dim=1, keepdim=True) == 0, torch.ones_like(valid), valid)
        return torch.multinomial(valid, 1).squeeze(1)
    
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
            
    def load_model_inference_only(self, path1, path2):
        for path, (net, target_net, opt, replay_memory, eps_attr, replayed_attr) in zip(
                (path1, path2), self._network_specs()):
            checkpoint = torch.load(path, weights_only=False)
            net.load_state_dict(checkpoint["dqn"])
            # target_net no se usa para decidir accion (solo turn_network/selection_network
            # se usan en inferencia), pero si mask_turn u otro sitio lo necesitara, cárgalo también.
            setattr(self, eps_attr, checkpoint["epsilon"])
            
    def save_model_inference_only(self, path1, path2):
        for path, (net, target_net, opt, replay_memory, eps_attr, replayed_attr) in zip(
                (path1, path2), self._network_specs()):
            torch.save({
                "dqn": net.state_dict(),
                "epsilon": getattr(self, eps_attr),
            }, path)
    
    def update_epsilon(self, n_games=1):
        decay_sel = EPSILON_SEL_DECAY ** n_games
        decay_turn = EPSILON_TURN_DECAY ** n_games
        self.epsilon_sel = max(EPSILON_SEL_MIN, self.epsilon_sel * decay_sel)
        self.epsilon_turn = max(EPSILON_TURN_MIN, self.epsilon_turn * decay_turn)
        
    def update_beta(self):
        self.replay_memory_sel.update_beta(self.replayed_selection)
        self.replay_memory_turn.update_beta(self.replayed_turn)

    def _optimize_step(self, loss, optimizer, network, target_network, replayed_counter_attr):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            replayed = getattr(self, replayed_counter_attr)
            if replayed % COPY_DQN == 0:
                target_network.load_state_dict(network.state_dict())
                
    def _multi_agent_double_dqn_target(self, batch, next_states, rewards, dones):
        with torch.no_grad():
            next_disposition = batch.next_types
            next_alive = batch.next_alive
            next_cooldowns = batch.next_cooldowns
            next_opp_disp = batch.next_opp_types

            next_masks = self.mask_turn(next_disposition, next_cooldowns, next_alive, next_opp_disp,
                                        torch.ones(len(next_states), 18, dtype=torch.bool))
            next_action_mask = (next_masks != float("-inf")).view(-1, 3, 6)                        

            next_qvalues_main = self.turn_network(next_states, action_mask=next_action_mask)        
            next_qvalues_main = next_qvalues_main.masked_fill(~next_masks, float("-inf"))
            next_q1 = next_qvalues_main[:, 0:6]
            next_q2 = next_qvalues_main[:, 6:12]
            next_q3 = next_qvalues_main[:, 12:18]
            next_a1 = next_q1.argmax(dim=1)
            next_a2 = next_q2.argmax(dim=1)
            next_a3 = next_q3.argmax(dim=1)
            next_actions = torch.stack([next_a1, next_a2 + 6, next_a3 + 12], dim=1)
            target_qvalues = self.target_turn_network(next_states, action_mask=next_action_mask)    
            next_qvalues = target_qvalues.gather(1, next_actions)
            next_warrior_mask = batch.next_alive
            next_qvalues = (next_qvalues * next_warrior_mask.float()).sum(dim=1)
            return rewards + DISCOUNT_FACTOR * next_qvalues * (~dones)
        
    @staticmethod
    def _environment_action_to_network(action):
        out = action.clone()
        out = torch.where(action == -1, torch.zeros_like(out), out)
        out = torch.where(action == 5, torch.full_like(out, 4), out)
        out = torch.where(action == 6, torch.full_like(out, 5), out)
        return out


    @staticmethod
    def _network_action_to_environment(action):
        if action == 4:
            return 5
        if action == 5:
            return 6
        return action