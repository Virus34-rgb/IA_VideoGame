"""
Agente DQN para Castle Game.
"""
import torch
from torch import nn
from typing import Tuple, Optional, Any, Dict, List

from AI.Agent.turnNetwork import TurnNetwork
from AI.Agent.selectionNetwork import SelectionNetwork
from AI.Agent.replayMemoryAN import ReplayMemoryAN
from AI.Agent.replayMemoryPM import ReplayMemoryPM
import constants


class PlayerAIV:
    def __init__(self, N: int, environment: Any, use_replay: bool = True) -> None:
        self.N: int = N
        self.environment: Any = environment
        self.name: str = "DqnPlayerV"

        self.epsilon_sel: float = constants.EPSILON_SELECTION
        self.epsilon_turn: float = constants.EPSILON_TURN
        self.epsilon_residual: float = constants.EPSILON_RESIDUAL
        
        selection_state_dim = constants.get_selection_state_dim()
        self.selection_network: SelectionNetwork = SelectionNetwork(sigma_init=constants.NOISY_SIGMA_INIT,input_size=selection_state_dim)
        self.target_selection_network: SelectionNetwork = SelectionNetwork(sigma_init=constants.NOISY_SIGMA_INIT,input_size=selection_state_dim)
        self.target_selection_network.load_state_dict(self.selection_network.state_dict())
        self.optimizer_sel = torch.optim.Adam(self.selection_network.parameters(), lr=constants.SELECTION_LEARNING_RATE)

        self.turn_network: TurnNetwork = TurnNetwork(sigma_init=constants.NOISY_SIGMA_INIT)
        self.target_turn_network: TurnNetwork = TurnNetwork(sigma_init=constants.NOISY_SIGMA_INIT)
        self.target_turn_network.load_state_dict(self.turn_network.state_dict())
        self.optimizer_turn = torch.optim.Adam(self.turn_network.parameters(), lr=constants.TURN_LEARNING_RATE)

        if use_replay:
            self.replay_memory_sel = ReplayMemoryPM(
                constants.SELECTION_REPLAY_DATA, state_dim=selection_state_dim,  
            )
            self.replay_memory_turn = ReplayMemoryAN(
                constants.TURN_REPLAY_DATA, state_dim=constants.TURN_STATE_DIM,             
            )
        else:
            self.replay_memory_sel = None
            self.replay_memory_turn = None

        self.replayed_selection: int = 0
        self.replayed_turn: int = 0
        self.elo = constants.ELO_INITIAL

    def remember_selection_batch(self, c_state, action, reward, next_c_state, done) -> None:
        self.replay_memory_sel.push_batch(c_state, action, reward, next_c_state, done)

    def remember_turn_batch(
        self, observation, action, reward, next_observation, done,
        alive, types, cooldowns, opp_types, next_types, next_alive, next_cooldowns, next_opp_types,
        instance_abilities, next_instance_abilities,   # NUEVO
    ) -> None:
        self.replay_memory_turn.push_batch(
            observation, action, reward, next_observation, done,
            alive, types, cooldowns, opp_types, next_types, next_alive, next_cooldowns, next_opp_types,
            instance_abilities, next_instance_abilities,   # NUEVO
        )

    def replay_selection(self) -> Optional[float]:
        self.selection_network.reset_noise()
        self.target_selection_network.reset_noise()

        if len(self.replay_memory_sel) < constants.BATCH_SIZE:
            return None

        self.replayed_selection += 1
        batch, tree_indices, weights = self.replay_memory_sel.sample(constants.BATCH_SIZE)
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
            target = rewards + constants.DISCOUNT_FACTOR * next_qvalues * (~dones)
            td_errors = torch.abs(q_selected - target)

        loss = self._loss_function(q_selected, target, weights)
        self._optimize_step(loss, self.optimizer_sel, self.selection_network, self.target_selection_network, "replayed_selection")
        self.replay_memory_sel.update_priorities(tree_indices, td_errors.detach().cpu().numpy())
        return loss.item()

    def replay_turn(self) -> Optional[float]:
        self.turn_network.reset_noise()
        self.target_turn_network.reset_noise()

        if len(self.replay_memory_turn) < constants.BATCH_SIZE:
            return None

        self.replayed_turn += 1
        batch, tree_indices, weights = self.replay_memory_turn.sample(constants.BATCH_SIZE)
        weights = torch.tensor(weights, dtype=torch.float32)

        states = batch.states
        warrior_mask = batch.alive
        next_states = batch.next_states
        rewards = batch.rewards
        dones = batch.dones

        actions_b = self._environment_action_to_network(batch.actions)

        current_mask_flat = self.mask_turn(
            batch.types, batch.cooldowns, batch.alive, batch.opp_types,
            batch.instance_abilities,   # NUEVO
            torch.ones(len(states), 18, dtype=torch.bool),
        )
        current_action_mask = (current_mask_flat != float("-inf")).view(-1, 3, 6)

        qvalues = self.turn_network(states, action_mask=current_action_mask)

        offsets = torch.tensor([0, 6, 12])
        actions_global = actions_b + offsets
        q_selected = qvalues.gather(1, actions_global)
        q_selected = q_selected * warrior_mask.float()
        q_selected = q_selected.sum(dim=1)

        target = self._multi_agent_double_dqn_target(batch, next_states, rewards, dones)

        with torch.no_grad():
            td_errors = torch.abs(q_selected - target)

        loss = self._loss_function(q_selected, target, weights)
        self._optimize_step(loss, self.optimizer_turn, self.turn_network, self.target_turn_network, "replayed_turn")
        self.replay_memory_turn.update_priorities(tree_indices, td_errors.detach().cpu().numpy())
        return loss.item()

    @staticmethod
    def _loss_function(input, target, weights):
        loss = nn.SmoothL1Loss(reduction="none")(input, target)
        return (loss * weights).mean()

    def selection(self, batch_encoded_states, disposition, opp_initial_warrior, castle_alive=None, already_used=None):
        """
        Modo catálogo (USE_META_GAME=False): elige (tipo 1..5, posición 0-2) libremente.
        Modo castillo (USE_META_GAME=True): elige (slot de castillo 0..MAX_CASTLE_SIZE-1, posición 0-2)
            libremente entre las instancias vivas y no usadas aún en este draft.
        """
        if constants.RESET_IN_DECISIONS:
            self.selection_network.reset_noise()

        states = batch_encoded_states.float()
        logits = self.selection_network(states)
        masked_logits = self._mask_selection(logits, disposition, castle_alive, already_used)

        greedy = torch.argmax(masked_logits, dim=1)
        random_action = self._random_valid_action(masked_logits)

        epsilon_efectivo = self.epsilon_residual if not self.selection_network.training else 0.0
        explora = (torch.rand(self.N) < epsilon_efectivo) | (opp_initial_warrior == 0)
        action = torch.where(explora, random_action, greedy)

        item_index = action // 3
        position = action % 3

        if not constants.USE_META_GAME:
            item_index = item_index + 1   # compatibilidad histórica: tipo 1..WARRIOR_QUANTITY

        return item_index, position, action

    def _mask_selection(self, logits, disposition, castle_alive=None, already_used=None):
        """
        Máscara genérica de selección libre: para cada (item, posición), es válida si
        el item está disponible Y la posición de combate está libre. Ni el item ni la
        posición se emparejan de forma fija — cualquier combinación válida es elegible.
        """
        N = disposition.shape[0]
        ocupado_pos = disposition > 0   # (N,3) posiciones de combate ya colocadas

        if constants.USE_META_GAME:
            num_items = constants.MAX_CASTLE_SIZE
            item_disponible = castle_alive & ~already_used   # (N, MAX_CASTLE_SIZE)
        else:
            num_items = constants.WARRIOR_QUANTITY
            usados_tipo = torch.zeros(N, constants.WARRIOR_QUANTITY, dtype=torch.bool)
            for slot in range(3):
                tipo_en_slot = disposition[:, slot]
                hay_tipo = tipo_en_slot > 0
                idx = (tipo_en_slot - 1).clamp(min=0)
                if hay_tipo.any():
                    rows = torch.arange(N)[hay_tipo]
                    usados_tipo[rows, idx[hay_tipo]] = True
            item_disponible = ~usados_tipo

        item_expand = item_disponible.unsqueeze(-1).expand(N, num_items, 3)
        pos_libre = (~ocupado_pos).unsqueeze(1).expand(N, num_items, 3)

        mask = (item_expand & pos_libre).reshape(N, num_items * 3)
        return logits.masked_fill(~mask, float("-inf"))

    def turn(self, batch_encoded_obs, own_disposition, own_cooldowns, own_alive, enemy_disposition, own_instance_abilities):
        # NUEVO parámetro own_instance_abilities
        if constants.RESET_IN_DECISIONS:
            self.turn_network.reset_noise()

        obs = batch_encoded_obs.float()
        logits = self.turn_network(obs)
        masked_logits = self.mask_turn(own_disposition, own_cooldowns, own_alive, enemy_disposition, own_instance_abilities, logits)

        masked_3d = masked_logits.view(self.N, 3, 6)
        hay_valida = (masked_3d != float("-inf")).any(dim=-1)

        greedy = torch.argmax(masked_3d, dim=-1)
        random_action = self._random_valid_action(masked_3d.reshape(self.N * 3, 6))
        random_action = random_action.view(self.N, 3)

        epsilon_efectivo = self.epsilon_residual if not self.turn_network.training else 0.0
        explora = torch.rand(self.N, 3) < epsilon_efectivo
        elegido = torch.where(explora, random_action, greedy)

        codigo = self._decode_ability_index(elegido)
        actions = torch.where(hay_valida, codigo, torch.full_like(codigo, -1))
        return actions

    def mask_turn(self, own_disposition, own_cooldowns, own_alive, enemy_disposition, own_instance_abilities, logits):
        # NUEVO parámetro own_instance_abilities
        N = own_disposition.shape[0]
        mask = own_alive.unsqueeze(-1).expand(N, 3, 6).clone()

        mask[:, :, :4] &= (own_cooldowns == 0)   # CAMBIADO: antes ~own_cooldowns (bool inválido con dtype long)

        # NUEVO: la tabla de target ahora tiene tamaño de POOL (no 4). Hay que seleccionar,
        # de las MAX_POOL_SIZE columnas, solo las 4 que están realmente equipadas por slot.
        table = self.environment.target_mask_por_tipo_habilidad          # (num_types, POOL, 3)
        target_mask_pool = table[own_disposition]                        # (N, 3, POOL, 3)
        idx = own_instance_abilities.unsqueeze(-1).expand(-1, -1, -1, 3)  # (N, 3, 4, 3)
        target_mask_full = target_mask_pool.gather(2, idx)                # (N, 3, 4, 3)

        enemy_ocupado = (enemy_disposition > 0).unsqueeze(1).unsqueeze(1)
        hay_target_valido = (target_mask_full & enemy_ocupado).any(dim=-1)
        sin_target = ~hay_target_valido & target_mask_full.any(dim=-1)

        mask[:, :, :4] &= ~sin_target

        mask[:, 0, 5] = False
        mask[:, 2, 4] = False

        mask_flat = mask.reshape(N, 18)
        return logits.masked_fill(~mask_flat, float("-inf"))

    @staticmethod
    def _random_valid_action(masked_logits):
        valid = (masked_logits != float("-inf")).float()
        valid = torch.where(valid.sum(dim=1, keepdim=True) == 0, torch.ones_like(valid), valid)
        return torch.multinomial(valid, 1).squeeze(1)

    @staticmethod
    def _decode_ability_index(idx_0_5):
        return torch.where(idx_0_5 == 4, torch.full_like(idx_0_5, 5), torch.where(idx_0_5 == 5, torch.full_like(idx_0_5, 6), idx_0_5))

    @staticmethod
    def _environment_action_to_network(action):
        out = action.clone()
        out = torch.where(action == -1, torch.zeros_like(out), out)
        out = torch.where(action == 5, torch.full_like(out, 4), out)
        out = torch.where(action == 6, torch.full_like(out, 5), out)
        return out

    @staticmethod
    def _network_action_to_environment(action: int) -> int:
        if action == 4:
            return 5
        if action == 5:
            return 6
        return action

    def update_epsilon(self, n_games: int = 1) -> None:
        decay_sel = constants.EPSILON_SEL_DECAY ** n_games
        decay_turn = constants.EPSILON_TURN_DECAY ** n_games
        self.epsilon_sel = max(constants.EPSILON_SEL_MIN, self.epsilon_sel * decay_sel)
        self.epsilon_turn = max(constants.EPSILON_TURN_MIN, self.epsilon_turn * decay_turn)

    def reset_noise(self):
        self.selection_network.reset_noise()
        self.turn_network.reset_noise()
        self.target_selection_network.reset_noise()
        self.target_turn_network.reset_noise()

    def update_beta(self) -> None:
        self.replay_memory_sel.update_beta(self.replayed_selection)
        self.replay_memory_turn.update_beta(self.replayed_turn)

    def _optimize_step(self, loss, optimizer, network, target_network, replayed_counter_attr):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        replayed = getattr(self, replayed_counter_attr)
        if replayed % constants.COPY_DQN == 0:
            target_network.load_state_dict(network.state_dict())

    def _multi_agent_double_dqn_target(self, batch, next_states, rewards, dones):
        with torch.no_grad():
            next_disposition = batch.next_types
            next_alive = batch.next_alive
            next_cooldowns = batch.next_cooldowns
            next_opp_disp = batch.next_opp_types

            next_masks = self.mask_turn(
                next_disposition, next_cooldowns, next_alive, next_opp_disp,
                batch.next_instance_abilities,   # NUEVO
                torch.ones(len(next_states), 18, dtype=torch.bool),
            )
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

            return rewards + (constants.DISCOUNT_FACTOR ** constants.N_STEP) * next_qvalues * (~dones)

    def _network_specs(self):
        return [
            (self.selection_network, self.target_selection_network, self.optimizer_sel, self.replay_memory_sel, "epsilon_sel", "replayed_selection"),
            (self.turn_network, self.target_turn_network, self.optimizer_turn, self.replay_memory_turn, "epsilon_turn", "replayed_turn"),
        ]

    def save_model(self, path1: str, path2: str) -> None:
        for path, (net, target_net, opt, replay_memory, eps_attr, replayed_attr) in zip((path1, path2), self._network_specs()):
            torch.save({
                "dqn": net.state_dict(), "targetdqn": target_net.state_dict(), "optimizer": opt.state_dict(),
                "epsilon": getattr(self, eps_attr), "replayed": getattr(self, replayed_attr),
                "replay_memory": replay_memory.state_dict(), "elo": self.elo,
            }, path)

    def load_model(self, path1: str, path2: str) -> None:
        for path, (net, target_net, opt, replay_memory, eps_attr, replayed_attr) in zip((path1, path2), self._network_specs()):
            checkpoint = torch.load(path, weights_only=False)
            net.load_state_dict(checkpoint["dqn"])
            target_net.load_state_dict(checkpoint["targetdqn"])
            opt.load_state_dict(checkpoint["optimizer"])
            setattr(self, eps_attr, checkpoint["epsilon"])
            setattr(self, replayed_attr, checkpoint["replayed"])
            replay_memory.load_state_dict(checkpoint["replay_memory"])
            self.elo = float(checkpoint.get("elo", constants.ELO_INITIAL))

    def load_model_inference_only(self, path1: str, path2: str) -> None:
        for path, (net, target_net, opt, replay_memory, eps_attr, replayed_attr) in zip((path1, path2), self._network_specs()):
            checkpoint = torch.load(path, weights_only=False)
            net.load_state_dict(checkpoint["dqn"])
            setattr(self, eps_attr, checkpoint["epsilon"])
            self.elo = float(checkpoint.get("elo", constants.ELO_INITIAL))

    def save_model_inference_only(self, path1: str, path2: str) -> None:
        for path, (net, target_net, opt, replay_memory, eps_attr, replayed_attr) in zip((path1, path2), self._network_specs()):
            torch.save({"dqn": net.state_dict(), "epsilon": getattr(self, eps_attr), "elo": self.elo}, path)