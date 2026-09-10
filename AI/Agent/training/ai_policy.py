"""
Decisión de acciones (draft y turno) para un agente PlayerAIV. Extraído de
PlayerAIV: selection, turn, _mask_selection, _random_valid_action,
compute_action_mask, reset_noise (de las redes de decisión).

Diseño: epsilon_sel/epsilon_turn/epsilon_residual SÍ viven aquí como self.X,
porque son estado de "cómo explora la política" -- consumido solo por esta
clase (turn()/selection()) y por update_epsilon(). No los necesita ni
AgentTrainer ni CheckpointManager directamente; CheckpointManager los recibe
como parámetro cuando PlayerAIV se los pasa explícitamente para guardar.
"""
import torch

from AI.Environment.action_mask import compute_action_mask
import constants


class AIPolicy:
    def __init__(self, N: int, environment, selection_network, turn_network):
        self.N = N
        self.environment = environment

        self.selection_network = selection_network
        self.turn_network = turn_network

        self.epsilon_sel: float = constants.EPSILON_SELECTION
        self.epsilon_turn: float = constants.EPSILON_TURN
        self.epsilon_residual: float = constants.EPSILON_RESIDUAL

    def selection(self, batch_encoded_states, disposition, opp_initial_warrior, castle_alive=None, already_used=None, castle_types=None):
        if constants.RESET_IN_DECISIONS:
            self.selection_network.reset_noise()

        states = batch_encoded_states.float()
        with torch.inference_mode():
            logits = self.selection_network(states)
        masked_logits = self._mask_selection(logits, disposition, castle_alive, already_used, castle_types)

        greedy = torch.argmax(masked_logits, dim=1)
        random_action = self._random_valid_action(masked_logits)

        epsilon_efectivo = self.epsilon_sel if self.selection_network.training else self.epsilon_residual
        explora = (torch.rand(self.N) < epsilon_efectivo) | (opp_initial_warrior == 0)
        action = torch.where(explora, random_action, greedy)

        item_index = action // 3
        position = action % 3
        return item_index, position, action

    def _mask_selection(self, logits, disposition, castle_alive=None, already_used=None, castle_types=None):
        N = disposition.shape[0]
        ocupado_pos = disposition > 0
        num_items = constants.MAX_CASTLE_SIZE
        item_disponible_base = castle_alive & ~already_used

        tipo_usado = torch.zeros(N, constants.WARRIOR_QUANTITY, dtype=torch.bool, device=disposition.device)
        for slot in range(3):
            tipo = disposition[:, slot]
            mask = (tipo > 0)
            idx = (tipo - 1).clamp(min=0)
            tipo_usado[mask, idx[mask]] = True

        tipos_slot = castle_alive * castle_types
        idx_tipo = (tipos_slot - 1).clamp(min=0)
        tipo_disponible = ~tipo_usado.gather(1, idx_tipo)
        item_disponible = item_disponible_base & tipo_disponible

        item_expand = item_disponible.unsqueeze(-1).expand(N, num_items, 3)
        pos_libre = (~ocupado_pos).unsqueeze(1).expand(N, num_items, 3)
        mask = (item_expand & pos_libre).reshape(N, num_items * 3)
        return logits.masked_fill(~mask, float("-inf"))

    def turn(self, batch_encoded_obs, own_disposition, own_cooldowns, own_alive, enemy_disposition, own_instance_abilities):
        if constants.RESET_IN_DECISIONS:
            self.turn_network.reset_noise()

        obs = batch_encoded_obs.float()
        action_mask = compute_action_mask(own_disposition, own_cooldowns, own_alive, 
                                               enemy_disposition, own_instance_abilities,self.environment.target_mask_por_tipo_habilidad)
        with torch.inference_mode():
            logits = self.turn_network(obs, action_mask=action_mask)
        masked_logits = logits.masked_fill(~action_mask.reshape(self.N, 18), float("-inf"))
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

    def turn_with_mask(self, batch_encoded_obs, own_disposition, own_cooldowns, own_alive, enemy_disposition,
                        own_instance_abilities, precomputed_action_mask):

        if constants.RESET_IN_DECISIONS:
            self.turn_network.reset_noise()

        obs = batch_encoded_obs.float()
        action_mask = precomputed_action_mask
        with torch.inference_mode():
            logits = self.turn_network(obs, action_mask=action_mask)
        masked_logits = logits.masked_fill(~action_mask.reshape(self.N, 18), float("-inf"))
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

    @staticmethod
    def _random_valid_action(masked_logits):
        valid = (masked_logits != float("-inf")).float()
        valid = torch.where(valid.sum(dim=1, keepdim=True) == 0, torch.ones_like(valid), valid)
        noise = torch.rand_like(valid)
        scored = noise * valid
        return torch.argmax(scored, dim=1)

    @staticmethod
    def _decode_ability_index(idx_0_5):
        return torch.where(idx_0_5 == 4, torch.full_like(idx_0_5, 5), torch.where(idx_0_5 == 5, torch.full_like(idx_0_5, 6), idx_0_5))

    def update_epsilon(self, n_games: int = 1) -> None:
        decay_sel = constants.EPSILON_SEL_DECAY ** n_games
        decay_turn = constants.EPSILON_TURN_DECAY ** n_games
        self.epsilon_sel = max(constants.EPSILON_SEL_MIN, self.epsilon_sel * decay_sel)
        self.epsilon_turn = max(constants.EPSILON_TURN_MIN, self.epsilon_turn * decay_turn)

    def reset_noise(self):
        self.selection_network.reset_noise()
        self.turn_network.reset_noise()