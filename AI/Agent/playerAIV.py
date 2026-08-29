"""
Agente DQN para Castle Game.

Contiene dos redes: una para la selección de equipo (SelectionNetwork)
y otra para la selección de acciones por turno (TurnNetwork).
Ambas usan Double DQN, PER, y N‑step returns.
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
    """
    Agente DQN para el Castle Game.

    Gestiona:
        - Selección de equipo (fase de draft)
        - Selección de acciones por turno (fase de batalla)
        - Replay de experiencias con PER (Prioritized Experience Replay)
        - Double DQN (red online + target)
        - Exploración ε-greedy (a futuro reemplazado por Noisy Networks)
    """

    def __init__(self, N: int, environment: Any,use_replay : bool = True) -> None:
        """
        Args:
            N: Número de partidas paralelas.
            environment: Instancia de VectorizedEnvironment (necesaria para tablas estáticas).
        """
        self.N: int = N
        self.environment: Any = environment
        self.name: str = "DqnPlayerV"

        # Parámetros de exploración
        self.epsilon_sel: float = constants.EPSILON_SELECTION
        self.epsilon_turn: float = constants.EPSILON_TURN
        self.epsilon_residual : float = constants.EPSILON_RESIDUAL

        # Red de selección
        self.selection_network: SelectionNetwork = SelectionNetwork(sigma_init=constants.NOISY_SIGMA_INIT)
        self.target_selection_network: SelectionNetwork = SelectionNetwork(sigma_init=constants.NOISY_SIGMA_INIT)
        self.target_selection_network.load_state_dict(self.selection_network.state_dict())
        self.optimizer_sel = torch.optim.Adam(
            self.selection_network.parameters(),
            lr=constants.SELECTION_LEARNING_RATE,
        )

        # Red de turno
        self.turn_network: TurnNetwork = TurnNetwork(sigma_init=constants.NOISY_SIGMA_INIT)
        self.target_turn_network: TurnNetwork = TurnNetwork(sigma_init=constants.NOISY_SIGMA_INIT)
        self.target_turn_network.load_state_dict(self.turn_network.state_dict())
        self.optimizer_turn = torch.optim.Adam(
            self.turn_network.parameters(),
            lr=constants.TURN_LEARNING_RATE,
        )
        if use_replay:
            self.replay_memory_sel = ReplayMemoryPM(
                constants.SELECTION_REPLAY_DATA,
                state_dim=46,  # Dimensión del estado de selección
            )
            self.replay_memory_turn = ReplayMemoryAN(
                constants.TURN_REPLAY_DATA,
                state_dim=58,  # Dimensión de la observación del turno
            )
        else:
            self.replay_memory_sel = None
            self.replay_memory_turn = None

        # Contadores de replays (para sincronización y beta)
        self.replayed_selection: int = 0
        self.replayed_turn: int = 0
        
        self.elo = constants.ELO_INITIAL

    def remember_selection_batch(
        self,
        c_state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_c_state: Optional[torch.Tensor],
        done: torch.Tensor,
    ) -> None:
        """Almacena un lote de experiencias de selección."""
        self.replay_memory_sel.push_batch(c_state, action, reward, next_c_state, done)

    def remember_turn_batch(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        done: torch.Tensor,
        alive: torch.Tensor,
        types: torch.Tensor,
        cooldowns: torch.Tensor,
        opp_types: torch.Tensor,
        next_types: torch.Tensor,
        next_alive: torch.Tensor,
        next_cooldowns: torch.Tensor,
        next_opp_types: torch.Tensor,
    ) -> None:
        """Almacena un lote de experiencias de turno."""
        self.replay_memory_turn.push_batch(
            observation,
            action,
            reward,
            next_observation,
            done,
            alive,
            types,
            cooldowns,
            opp_types,
            next_types,
            next_alive,
            next_cooldowns,
            next_opp_types,
        )

    def replay_selection(self) -> Optional[float]:
        """
        Realiza un paso de replay para la red de selección (Double DQN + PER).

        Returns:
            Pérdida del paso, o None si no hay suficientes experiencias.
        """
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

        # Q-values actuales
        qvalues = self.selection_network(states)
        q_selected = qvalues.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target (Double DQN)
        with torch.no_grad():
            next_actions = self.selection_network(next_states).argmax(dim=1)
            next_qvalues = self.target_selection_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target = rewards + constants.DISCOUNT_FACTOR * next_qvalues * (~dones)
            td_errors = torch.abs(q_selected - target)

        # Pérdida con pesos de importancia
        loss = self._loss_function(q_selected, target, weights)

        # Optimizar y actualizar prioridades
        self._optimize_step(
            loss,
            self.optimizer_sel,
            self.selection_network,
            self.target_selection_network,
            "replayed_selection",
        )
        self.replay_memory_sel.update_priorities(
            tree_indices,
            td_errors.detach().cpu().numpy(),
        )

        return loss.item()

    def replay_turn(self) -> Optional[float]:
        """
        Realiza un paso de replay para la red de turno (Double DQN + PER + N‑step).

        Returns:
            Pérdida del paso, o None si no hay suficientes experiencias.
        """
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

        # Convertir acciones de entorno a índices de red
        actions_b = self._environment_action_to_network(batch.actions)

        # Máscara de acciones válidas para el estado actual
        current_mask_flat = self.mask_turn(
            batch.types,
            batch.cooldowns,
            batch.alive,
            batch.opp_types,
            torch.ones(len(states), 18, dtype=torch.bool),
        )
        current_action_mask = (current_mask_flat != float("-inf")).view(-1, 3, 6)

        # Q-values actuales (con máscara)
        qvalues = self.turn_network(states, action_mask=current_action_mask)

        # Seleccionar Q para las acciones tomadas (por slot)
        offsets = torch.tensor([0, 6, 12])
        actions_global = actions_b + offsets
        q_selected = qvalues.gather(1, actions_global)
        q_selected = q_selected * warrior_mask.float()
        q_selected = q_selected.sum(dim=1)

        # Target (Double DQN con N‑step)
        target = self._multi_agent_double_dqn_target(batch, next_states, rewards, dones)

        with torch.no_grad():
            td_errors = torch.abs(q_selected - target)

        # Pérdida con pesos de importancia
        loss = self._loss_function(q_selected, target, weights)

        # Optimizar y actualizar prioridades
        self._optimize_step(
            loss,
            self.optimizer_turn,
            self.turn_network,
            self.target_turn_network,
            "replayed_turn",
        )
        self.replay_memory_turn.update_priorities(
            tree_indices,
            td_errors.detach().cpu().numpy(),
        )

        return loss.item()

    @staticmethod
    def _loss_function(input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Pérdida Smooth L1 ponderada por importancia de muestreo (PER).
        """
        loss = nn.SmoothL1Loss(reduction="none")(input, target)
        return (loss * weights).mean()

    def selection(
        self,
        batch_encoded_states: torch.Tensor,
        disposition: torch.Tensor,
        opp_initial_warrior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Selecciona un guerrero para cada partida en la fase de draft.

        Args:
            batch_encoded_states: (N, dim) estados codificados de selección.
            disposition: (N, 3) disposición actual (guerreros ya colocados).
            opp_initial_warrior: (N,) ID del primer guerrero del oponente (0 si ninguno).

        Returns:
            Tupla (warrior_id, position, action_index):
                - warrior_id: (N,) ID del guerrero seleccionado (1..WARRIOR_QUANTITY)
                - position: (N,) posición donde colocarlo (0-2)
                - action_index: (N,) índice de acción usado (para guardar en replay)
        """
        if(constants.RESET_IN_DECISIONS):
            self.selection_network.reset_noise()

        states = batch_encoded_states.float()
        logits = self.selection_network(states)
        masked_logits = self._mask_selection(logits, disposition)

        # Acción greedy
        greedy = torch.argmax(masked_logits, dim=1)

        # Acción aleatoria válida
        random_action = self._random_valid_action(masked_logits)

        # Exploración: si el oponente no ha seleccionado nada (primer paso), forzar exploración
        epsilon_efectivo = self.epsilon_residual if not self.selection_network.training else 0.0
        explora = (torch.rand(self.N) < epsilon_efectivo) | (opp_initial_warrior == 0)
        action = torch.where(explora, random_action, greedy)

        warrior_index = action // 3
        position = action % 3

        # +1 porque los IDs de guerrero empiezan en 1
        return warrior_index + 1, position, action

    def _mask_selection(self, logits: torch.Tensor, disposition: torch.Tensor) -> torch.Tensor:
        """
        Enmascara acciones inválidas en la selección de equipo.

        Acciones inválidas:
            - Colocar un guerrero ya colocado.
            - Colocar en una posición ya ocupada.
        """
        N = disposition.shape[0]
        mask = torch.ones(N, constants.WARRIOR_QUANTITY * 3, dtype=torch.bool)

        ocupado = disposition > 0
        warrior_idx = (disposition - 1).clamp(min=0)

        # Invalida las 3 acciones del guerrero ya colocado
        accion_base = warrior_idx * 3
        for offset in range(3):
            accion = accion_base + offset
            mask.scatter_(
                1,
                accion,
                torch.where(
                    ocupado,
                    torch.zeros_like(accion, dtype=torch.bool),
                    mask.gather(1, accion),
                ),
            )

        # Invalida acciones que colocarían en un slot ya ocupado
        for slot in range(3):
            slot_ocupado = ocupado[:, slot]
            for wi in range(constants.WARRIOR_QUANTITY):
                accion = torch.full((N, 1), wi * 3 + slot, dtype=torch.long)
                mask.scatter_(
                    1,
                    accion,
                    torch.where(
                        slot_ocupado.unsqueeze(1),
                        torch.zeros_like(accion, dtype=torch.bool),
                        mask.gather(1, accion),
                    ),
                )

        return logits.masked_fill(~mask, float("-inf"))

    def turn(
        self,
        batch_encoded_obs: torch.Tensor,
        own_disposition: torch.Tensor,
        own_cooldowns: torch.Tensor,
        own_alive: torch.Tensor,
        enemy_disposition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Selecciona acciones para todos los guerreros vivos en el turno actual.

        Args:
            batch_encoded_obs: (N, dim) observaciones del turno.
            own_disposition: (N, 3) IDs de los guerreros propios.
            own_cooldowns: (N, 3, 4) cooldowns de habilidades.
            own_alive: (N, 3) máscara de guerreros vivos.
            enemy_disposition: (N, 3) IDs de los guerreros enemigos.

        Returns:
            actions: (N, 3) acciones elegidas (0-3 habilidad, 5=movPos, 6=movNeg, -1 si muerto).
        """
        if(constants.RESET_IN_DECISIONS):
            self.turn_network.reset_noise()

        obs = batch_encoded_obs.float()
        logits = self.turn_network(obs)
        masked_logits = self.mask_turn(
            own_disposition,
            own_cooldowns,
            own_alive,
            enemy_disposition,
            logits,
        )

        masked_3d = masked_logits.view(self.N, 3, 6)
        hay_valida = (masked_3d != float("-inf")).any(dim=-1)

        # Acción greedy por slot
        greedy = torch.argmax(masked_3d, dim=-1)

        # Acción aleatoria válida
        random_action = self._random_valid_action(masked_3d.reshape(self.N * 3, 6))
        random_action = random_action.view(self.N, 3)

        # Exploración ε-greedy por slot
        epsilon_efectivo = self.epsilon_residual if not self.turn_network.training else 0.0
        explora = torch.rand(self.N, 3) < epsilon_efectivo
        elegido = torch.where(explora, random_action, greedy)

        # Convertir índice de red (0-5) a código de entorno (0-3,5,6)
        codigo = self._decode_ability_index(elegido)
        actions = torch.where(hay_valida, codigo, torch.full_like(codigo, -1))

        return actions

    def mask_turn(
        self,
        own_disposition: torch.Tensor,
        own_cooldowns: torch.Tensor,
        own_alive: torch.Tensor,
        enemy_disposition: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enmascara acciones inválidas en el turno.

        Acciones inválidas:
            - Guerrero muerto → todas sus acciones inválidas.
            - Habilidad en cooldown → inválida.
            - Habilidad sin objetivo válido (ej. ataque a distancia sin enemigo vivo en esa posición).
            - Movimiento desde posiciones extremas.
        """
        N = own_disposition.shape[0]
        mask = own_alive.unsqueeze(-1).expand(N, 3, 6).clone()

        # Cooldowns: desactivar habilidades en enfriamiento
        mask[:, :, :4] &= ~own_cooldowns

        # Objetivos: desactivar habilidades sin enemigo alcanzable
        table = self.environment.target_mask_por_tipo_habilidad
        target_mask_full = table[own_disposition]  # (N, 3, 4, 3)

        enemy_ocupado = (enemy_disposition > 0).unsqueeze(1).unsqueeze(1)
        hay_target_valido = (target_mask_full & enemy_ocupado).any(dim=-1)
        sin_target = ~hay_target_valido & target_mask_full.any(dim=-1)

        mask[:, :, :4] &= ~sin_target

        # Movimientos inválidos en extremos
        mask[:, 0, 5] = False   # movNeg inválido en slot 0
        mask[:, 2, 4] = False   # movPos inválido en slot 2

        mask_flat = mask.reshape(N, 18)
        return logits.masked_fill(~mask_flat, float("-inf"))

    @staticmethod
    def _random_valid_action(masked_logits: torch.Tensor) -> torch.Tensor:
        """
        Selecciona una acción aleatoria entre las válidas (donde logits != -inf).
        """
        valid = (masked_logits != float("-inf")).float()
        # Si no hay acciones válidas, uniforme sobre todas (caso extremo)
        valid = torch.where(
            valid.sum(dim=1, keepdim=True) == 0,
            torch.ones_like(valid),
            valid,
        )
        return torch.multinomial(valid, 1).squeeze(1)

    @staticmethod
    def _decode_ability_index(idx_0_5: torch.Tensor) -> torch.Tensor:
        """
        Convierte índice de red (0-5) a código de entorno (0-3,5,6).
        """
        return torch.where(
            idx_0_5 == 4,
            torch.full_like(idx_0_5, 5),
            torch.where(idx_0_5 == 5, torch.full_like(idx_0_5, 6), idx_0_5),
        )

    @staticmethod
    def _environment_action_to_network(action: torch.Tensor) -> torch.Tensor:
        """
        Convierte acción de entorno (0-3,5,6) a índice de red (0-5).
        """
        out = action.clone()
        out = torch.where(action == -1, torch.zeros_like(out), out)
        out = torch.where(action == 5, torch.full_like(out, 4), out)
        out = torch.where(action == 6, torch.full_like(out, 5), out)
        return out

    @staticmethod
    def _network_action_to_environment(action: int) -> int:
        """Convierte acción de red (0-5) a acción de entorno (0-3,5,6)."""
        if action == 4:
            return 5
        if action == 5:
            return 6
        return action

    def update_epsilon(self, n_games: int = 1) -> None:
        """Decae epsilon según las constantes."""
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
        """Actualiza el factor beta de PER para ambas memorias."""
        self.replay_memory_sel.update_beta(self.replayed_selection)
        self.replay_memory_turn.update_beta(self.replayed_turn)

    def _optimize_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        network: nn.Module,
        target_network: nn.Module,
        replayed_counter_attr: str,
    ) -> None:
        """
        Realiza un paso de optimización y sincroniza la target network si toca.
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        replayed = getattr(self, replayed_counter_attr)
        if replayed % constants.COPY_DQN == 0:
            target_network.load_state_dict(network.state_dict())

    def _multi_agent_double_dqn_target(
        self,
        batch: Any,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcula el target para Double DQN con N‑step en un entorno multi‑agente (3 slots).

        Cada slot es un agente independiente, pero comparten la misma red.
        """
        with torch.no_grad():
            next_disposition = batch.next_types
            next_alive = batch.next_alive
            next_cooldowns = batch.next_cooldowns
            next_opp_disp = batch.next_opp_types

            # Máscara para el siguiente estado
            next_masks = self.mask_turn(
                next_disposition,
                next_cooldowns,
                next_alive,
                next_opp_disp,
                torch.ones(len(next_states), 18, dtype=torch.bool),
            )
            next_action_mask = (next_masks != float("-inf")).view(-1, 3, 6)

            # Q-values de la red main para el siguiente estado
            next_qvalues_main = self.turn_network(next_states, action_mask=next_action_mask)
            next_qvalues_main = next_qvalues_main.masked_fill(~next_masks, float("-inf"))

            # Seleccionar mejor acción por slot (usando red main)
            next_q1 = next_qvalues_main[:, 0:6]
            next_q2 = next_qvalues_main[:, 6:12]
            next_q3 = next_qvalues_main[:, 12:18]
            next_a1 = next_q1.argmax(dim=1)
            next_a2 = next_q2.argmax(dim=1)
            next_a3 = next_q3.argmax(dim=1)
            next_actions = torch.stack([next_a1, next_a2 + 6, next_a3 + 12], dim=1)

            # Q-values de la target network para las acciones seleccionadas
            target_qvalues = self.target_turn_network(next_states, action_mask=next_action_mask)
            next_qvalues = target_qvalues.gather(1, next_actions)

            # Sumar Q-values de los slots vivos
            next_warrior_mask = batch.next_alive
            next_qvalues = (next_qvalues * next_warrior_mask.float()).sum(dim=1)

            # Target con N‑step descuento
            return rewards + (constants.DISCOUNT_FACTOR ** constants.N_STEP) * next_qvalues * (~dones)

    def _network_specs(self) -> List[Tuple[nn.Module, nn.Module, torch.optim.Optimizer, Any, str, str]]:
        """Devuelve las especificaciones de ambas redes para guardado/carga."""
        return [
            (
                self.selection_network,
                self.target_selection_network,
                self.optimizer_sel,
                self.replay_memory_sel,
                "epsilon_sel",
                "replayed_selection",
            ),
            (
                self.turn_network,
                self.target_turn_network,
                self.optimizer_turn,
                self.replay_memory_turn,
                "epsilon_turn",
                "replayed_turn",
            ),
        ]

    def save_model(self, path1: str, path2: str) -> None:
        """Guarda el estado completo (redes, optimizadores, buffer, etc.)."""
        for path, (net, target_net, opt, replay_memory, eps_attr, replayed_attr) in zip(
            (path1, path2), self._network_specs()
        ):
            torch.save(
                {
                    "dqn": net.state_dict(),
                    "targetdqn": target_net.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epsilon": getattr(self, eps_attr),
                    "replayed": getattr(self, replayed_attr),
                    "replay_memory": replay_memory.state_dict(),
                    "elo": self.elo,   # ← NUEVO, 
                },
                path,
            )

    def load_model(self, path1: str, path2: str) -> None:
        """Carga el estado completo."""
        for path, (net, target_net, opt, replay_memory, eps_attr, replayed_attr) in zip(
            (path1, path2), self._network_specs()
        ):
            checkpoint = torch.load(path, weights_only=False)
            net.load_state_dict(checkpoint["dqn"])
            target_net.load_state_dict(checkpoint["targetdqn"])
            opt.load_state_dict(checkpoint["optimizer"])
            setattr(self, eps_attr, checkpoint["epsilon"])
            setattr(self, replayed_attr, checkpoint["replayed"])
            replay_memory.load_state_dict(checkpoint["replay_memory"])
            self.elo = float(checkpoint.get("elo", constants.ELO_INITIAL))

    def load_model_inference_only(self, path1: str, path2: str) -> None:
        """Carga solo las redes (sin optimizador ni buffer) para inferencia."""
        for path, (net, target_net, opt, replay_memory, eps_attr, replayed_attr) in zip(
            (path1, path2), self._network_specs()
        ):
            checkpoint = torch.load(path, weights_only=False)
            net.load_state_dict(checkpoint["dqn"])
            # target_net no es necesario para inferencia, pero se carga por si acaso
            setattr(self, eps_attr, checkpoint["epsilon"])
            self.elo = float(checkpoint.get("elo", constants.ELO_INITIAL))

    def save_model_inference_only(self, path1: str, path2: str) -> None:
        """Guarda solo las redes (sin optimizador ni buffer) para la pool."""
        for path, (net, target_net, opt, replay_memory, eps_attr, replayed_attr) in zip(
            (path1, path2), self._network_specs()
        ):
            torch.save(
                {
                    "dqn": net.state_dict(),
                    "epsilon": getattr(self, eps_attr),
                    "elo": self.elo,   # ← NUEVO
                },
                path,
            )