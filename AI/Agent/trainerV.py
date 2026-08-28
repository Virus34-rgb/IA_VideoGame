"""
Entrenador para Castle Game.

Orquesta el entrenamiento y evaluación de agentes DQN en un entorno vectorizado.
Gestiona la recolección de experiencias, el replay, la pool de oponentes y el logging.
"""
import os
import time
from typing import Optional, Any, Tuple, Dict

import torch
import constants

from AI.Agent.choose_state import ChooseStateV
from AI.Agent.nstep_buffer import NStepBuffer
from AI.Agent.observationV import ObservationV


class TrainerV:
    """
    Entrenador para dos jugadores (P1 y P2) en el entorno vectorizado.

    Soporta:
    - Entrenamiento con self-play (P1 y P2 aprenden).
    - Evaluación con epsilon bajo.
    - Pool de oponentes para P2 (muestreo aleatorio o por habilidad en el futuro).
    - N‑step returns.
    - Logging de métricas y snapshots.
    """

    def __init__(
        self,
        player1: Any,
        player2: Any,
        environment: Any,
        opponent_pool: Any,
        train_batches: int,
        eval_batches: int,
        pathp1_1: str,
        pathp1_2: str,
        pathp2_1: str,
        pathp2_2: str,
        path_stats: str,
        path_stats2: str,
        logger: Optional[Any] = None,
        snapshot_every: int = 1000,
        progress_every: int = 1,
    ) -> None:
        """
        Args:
            player1: Agente P1 (normalmente el que se entrena).
            player2: Agente P2 (puede ser el mismo o un oponente de la pool).
            environment: Instancia de VectorizedEnvironment.
            opponent_pool: Instancia de OpponentPoolV.
            train_batches: Número de lotes de entrenamiento.
            eval_batches: Número de lotes de evaluación.
            pathp1_1, pathp1_2: Rutas para guardar/cargar modelos de P1 (selection y turn).
            pathp2_1, pathp2_2: Rutas para P2.
            path_stats, path_stats2: Rutas para estadísticas de entrenamiento/evaluación.
            logger: Instancia de MetricsLogger (opcional).
            snapshot_every: Cada cuántos lotes guardar snapshot.
            progress_every: Cada cuántos lotes mostrar progreso.
        """
        self.player1 = player1
        self.player2 = player2
        self.environment = environment
        self.N = environment.N
        self.opponent_pool = opponent_pool

        self.train_batches = train_batches
        self.eval_batches = eval_batches

        self.pathp1_1, self.pathp1_2 = pathp1_1, pathp1_2
        self.pathp2_1, self.pathp2_2 = pathp2_1, pathp2_2
        self.path_stats, self.path_stats2 = path_stats, path_stats2

        self.logger = logger
        self.snapshot_every = snapshot_every
        self.progress_every = progress_every

        # Catálogo de guerreros para codificación de selección
        self._catalog_ids = torch.arange(1, constants.WARRIOR_QUANTITY + 1, dtype=torch.long)

        # Estado de asignación de oponentes de la pool (se actualiza periódicamente)
        self._opponent_from_pool_mask = torch.zeros(self.N, dtype=torch.bool)
        self._grouped_opponents: Dict[int, Tuple[Any, torch.Tensor]] = {}

    def train(self) -> None:
        """Ejecuta el entrenamiento completo (carga de modelos, entrenamiento y guardado)."""
        self._load_if_exists()
        self._run(
            batches=self.train_batches,
            epsilon_turn=0.5,
            epsilon_sel=None,
            learn_p1=True,
            learn_p2=True,
            stats_path=self.path_stats,
            restore_epsilon=True,
        )
        self._save_if_supported(self.player1, self.pathp1_1, self.pathp1_2)
        self._save_if_supported(self.player2, self.pathp2_1, self.pathp2_2)

    def evaluate(self) -> None:
        """Ejecuta la evaluación (sin aprendizaje)."""
        self.environment.stats.reset()
        self._load_if_exists()
        self._run(
            batches=self.eval_batches,
            epsilon_turn=0.02,
            epsilon_sel=0.02,
            learn_p1=False,
            learn_p2=False,
            stats_path=self.path_stats2,
            restore_epsilon=True,
        )

    def _run(
        self,
        batches: int,
        epsilon_turn: float,
        epsilon_sel: Optional[float],
        learn_p1: bool,
        learn_p2: bool,
        stats_path: str,
        restore_epsilon: bool,
    ) -> None:
        """
        Bucle principal para entrenamiento o evaluación.

        Args:
            batches: Número de lotes a ejecutar.
            epsilon_turn: Epsilon para la política de turno.
            epsilon_sel: Epsilon para la política de selección (None = no cambiar).
            learn_p1, learn_p2: Si se debe aprender (replay).
            stats_path: Ruta donde guardar estadísticas.
            restore_epsilon: Si restaurar los epsilons originales al final.
        """
        save_every = max(1, int(batches * constants.SAVE_MODEL_FRACTION))
        pool_every = max(1, int(batches * constants.POOL_RANGE_FRACTION))
        snapshot_every = max(1, batches // 50)

        backup = self._set_epsilons(epsilon_turn, epsilon_sel)
        start_time = time.time()

        # P2 puede ser reemplazado por oponentes de la pool durante el entrenamiento
        p2_training_player = self.player2
        self._opponent_from_pool_mask = torch.zeros(self.N, dtype=torch.bool)
        self._grouped_opponents = {}

        for batch_idx in range(batches):
            # Guardar snapshot de P2 en la pool (periódicamente)
            if batch_idx != 0 and batch_idx % save_every == 0:
                self.opponent_pool.save_version(p2_training_player)

            # Actualizar asignación de oponentes de la pool (periódicamente)
            if batch_idx != 0 and batch_idx % pool_every == 0:
                from_pool, checkpoint_idx = self.opponent_pool.sample_assignment(
                    self.N, constants.POOL_PORCENTAGE
                )
                self._opponent_from_pool_mask = from_pool
                self._grouped_opponents = (
                    self.opponent_pool.build_grouped_opponents(
                        checkpoint_idx,
                        self.player1.__class__,
                        self.N,
                        self.environment,
                    )
                    if from_pool.any()
                    else {}
                )

            # Ejecutar un lote
            self._run_batch(batch_idx, learn_p1, learn_p2, p2_training_player)

            # Logging de snapshot
            if self.logger and (learn_p1 or learn_p2) and snapshot_every and batch_idx % snapshot_every == 0:
                self.logger.log_snapshot(
                    batch_idx,
                    self.player1,
                    p2_training_player,
                    self.environment.stats,
                )

            # Mostrar progreso
            self._print_progress(batch_idx, batches, start_time)

        if batches > 0:
            print()

        # Guardar estadísticas finales
        self.environment.stats.guardar_stats(stats_path, self.environment.warriors_classes)

        if restore_epsilon:
            self._restore_epsilons(backup)
            
    def _run_batch(
        self,
        batch_idx: int,
        learn_p1: bool,
        learn_p2: bool,
        p2_training_player: Any,
    ) -> None:
        """
        Ejecuta un lote completo: selección de equipos, turnos, recolección y replay.
        """
        # 1. Selección de equipos
        self.environment.reset()
        selection_states_p1, selection_actions_p1, selection_states_p2, selection_actions_p2 = (
            self._select_teams(p2_training_player)
        )

        # 2. Inicializar buffers y observaciones
        obs1_tensor, obs2_tensor = self._build_observations()
        reward1_acum = torch.zeros(self.N)
        reward2_acum = torch.zeros(self.N)

        n_steps_buffer_p1 = NStepBuffer(n_step=constants.N_STEP, gamma=constants.DISCOUNT_FACTOR)
        n_steps_buffer_p2 = NStepBuffer(n_step=constants.N_STEP, gamma=constants.DISCOUNT_FACTOR)

        # 3. Bucle de turnos
        while not self.environment.ended.all():
            self._run_turn(
                obs1_tensor,
                obs2_tensor,
                n_steps_buffer_p1,
                n_steps_buffer_p2,
                p2_training_player,
                learn_p1,
                learn_p2,
            )
            obs1_tensor, obs2_tensor = self._build_observations()
            reward1_acum += self._last_reward1
            reward2_acum += self._last_reward2

        # 4. Flush de experiencias pendientes en N‑step buffers
        if learn_p1:
            for experience in n_steps_buffer_p1.flush():
                self._remember_turn_batch(self.player1, experience)
        if learn_p2:
            for experience in n_steps_buffer_p2.flush():
                self._remember_turn_batch(
                    p2_training_player,
                    experience,
                    skip_mask=self._opponent_from_pool_mask,
                )

        # 5. Replay de turno y selección para P1
        if learn_p1:
            self._replay_turn_and_selection(
                self.player1,
                selection_states_p1,
                selection_actions_p1,
                reward1_acum,
                "p1",
                batch_idx,
            )
            if self.train_batches != 0:
                self.player1.update_beta()
                self.player1.update_epsilon(n_games=self.N)

        # 6. Replay de turno y selección para P2
        if learn_p2:
            self._replay_turn_and_selection(
                p2_training_player,
                selection_states_p2,
                selection_actions_p2,
                reward2_acum,
                "p2",
                batch_idx,
                skip_mask=self._opponent_from_pool_mask,
            )
            if self.train_batches != 0:
                p2_training_player.update_beta()
                p2_training_player.update_epsilon(n_games=self.N)

        # 7. Acumular rewards totales (para estadísticas)
        self.environment.stats.total_reward_p1 += reward1_acum.sum().item()
        self.environment.stats.total_reward_p2 += reward2_acum.sum().item()

    def _run_turn(
        self,
        obs1_tensor: torch.Tensor,
        obs2_tensor: torch.Tensor,
        n_steps_buffer_p1: NStepBuffer,
        n_steps_buffer_p2: NStepBuffer,
        p2_training_player: Any,
        learn_p1: bool,
        learn_p2: bool,
    ) -> None:
        """
        Ejecuta un solo turno: obtiene acciones, aplica el turno y guarda experiencias.
        """
        # Guardar estado actual para N‑step
        p1_alive_now = self.environment.p1_alive
        p2_alive_now = self.environment.p2_alive
        p1_types_now = self.environment.p1_disposition
        p1_cd_now = self.environment.p1_cooldowns
        p2_types_now = self.environment.p2_disposition
        p2_cd_now = self.environment.p2_cooldowns
        p1_opp_types_now = self.environment.p2_disposition  # para P1, el oponente es P2
        p2_opp_types_now = self.environment.p1_disposition  # para P2, el oponente es P1

        # Obtener acciones
        action_p1 = self.player1.turn(
            obs1_tensor,
            self.environment.p1_disposition,
            self.environment.p1_cooldowns,
            self.environment.p1_alive,
            self.environment.p2_disposition,
        )
        action_p2 = self._turn_mixed_opponent(
            obs2_tensor,
            self._opponent_from_pool_mask,
            self._grouped_opponents,
            p2_training_player,
        )

        # Ejecutar turno en el entorno
        state, reward1, reward2, ended = self.environment.turn(action_p1, action_p2)

        # Guardar recompensas para acumulación posterior
        self._last_reward1 = reward1
        self._last_reward2 = reward2


        # Almacenar experiencias en buffers N‑step
        if learn_p1:
            exp_p1 = n_steps_buffer_p1.push(
                obs1_tensor,
                action_p1,
                reward1,
                ended,
                p1_alive_now,
                p1_types_now,
                p1_cd_now,
                p1_opp_types_now,
            )
            if exp_p1 is not None:
                self._remember_turn_batch(self.player1, exp_p1)

        if learn_p2:
            exp_p2 = n_steps_buffer_p2.push(
                obs2_tensor,
                action_p2,
                reward2,
                ended,
                p2_alive_now,
                p2_types_now,
                p2_cd_now,
                p2_opp_types_now,
            )
            if exp_p2 is not None:
                self._remember_turn_batch(
                    p2_training_player,
                    exp_p2,
                    skip_mask=self._opponent_from_pool_mask,
                )

    def _select_teams(self, p2_training_player: Any) -> Tuple[Tuple, Tuple, Tuple, Tuple]:
        """
        Realiza las 3 selecciones de equipo para P1 y P2.

        Returns:
            Tuple con:
                - selection_states_p1: (c1, c2, c3) para cada paso de P1.
                - selection_actions_p1: (a1, a2, a3) acciones de P1.
                - selection_states_p2: (c1, c2, c3) para cada paso de P2.
                - selection_actions_p2: (a1, a2, a3) acciones de P2.
        """
        # Paso 1: primer guerrero (sin información previa)
        cstate1_1 = self._encode_choose_batch(
            self.environment.p1_disposition,
            torch.zeros(self.N, dtype=torch.long),
            torch.zeros(self.N, dtype=torch.long),
        )
        cstate2_1 = self._encode_choose_batch(
            self.environment.p2_disposition,
            torch.zeros(self.N, dtype=torch.long),
            torch.zeros(self.N, dtype=torch.long),
        )

        warr1_1, pos1_1, action1_1 = self.player1.selection(
            cstate1_1,
            self.environment.p1_disposition,
            torch.zeros(self.N, dtype=torch.long),
        )
        warr2_1, pos2_1, action2_1 = p2_training_player.selection(
            cstate2_1,
            self.environment.p2_disposition,
            torch.zeros(self.N, dtype=torch.long),
        )

        # Colocar primer guerrero
        health1 = self.environment.max_health_por_tipo[warr1_1]
        health2 = self.environment.max_health_por_tipo[warr2_1]
        self.environment.team_selection(
            warr1_1, pos1_1, warr2_1, pos2_1,
            selected=0, health1=health1, health2=health2,
        )

        # Paso 2: segundo guerrero (conociendo el primero del oponente)
        cstate1_2 = self._encode_choose_batch(
            self.environment.p1_disposition,
            warr2_1,
            pos2_1 + 1,
        )
        cstate2_2 = self._encode_choose_batch(
            self.environment.p2_disposition,
            warr1_1,
            pos1_1 + 1,
        )

        warr1_2, pos1_2, action1_2 = self.player1.selection(
            cstate1_2,
            self.environment.p1_disposition,
            warr2_1,
        )
        warr2_2, pos2_2, action2_2 = p2_training_player.selection(
            cstate2_2,
            self.environment.p2_disposition,
            warr1_1,
        )

        health1 = self.environment.max_health_por_tipo[warr1_2]
        health2 = self.environment.max_health_por_tipo[warr2_2]
        self.environment.team_selection(
            warr1_2, pos1_2, warr2_2, pos2_2,
            selected=1, health1=health1, health2=health2,
        )

        # Paso 3: tercer guerrero (conociendo el primero del oponente)
        cstate1_3 = self._encode_choose_batch(
            self.environment.p1_disposition,
            warr2_1,
            pos2_1 + 1,
        )
        cstate2_3 = self._encode_choose_batch(
            self.environment.p2_disposition,
            warr1_1,
            pos1_1 + 1,
        )

        warr1_3, pos1_3, action1_3 = self.player1.selection(
            cstate1_3,
            self.environment.p1_disposition,
            warr2_1,
        )
        warr2_3, pos2_3, action2_3 = p2_training_player.selection(
            cstate2_3,
            self.environment.p2_disposition,
            warr1_1,
        )

        health1 = self.environment.max_health_por_tipo[warr1_3]
        health2 = self.environment.max_health_por_tipo[warr2_3]
        self.environment.team_selection(
            warr1_3, pos1_3, warr2_3, pos2_3,
            selected=2, health1=health1, health2=health2,
        )

        # Empaquetar estados y acciones
        selection_states_p1 = (cstate1_1, cstate1_2, cstate1_3)
        selection_actions_p1 = (action1_1, action1_2, action1_3)
        selection_states_p2 = (cstate2_1, cstate2_2, cstate2_3)
        selection_actions_p2 = (action2_1, action2_2, action2_3)

        return (
            selection_states_p1,
            selection_actions_p1,
            selection_states_p2,
            selection_actions_p2,
        )

    def _encode_choose_batch(
        self,
        disposition: torch.Tensor,
        opp_initial_warrior: torch.Tensor,
        opp_initial_position: torch.Tensor,
    ) -> torch.Tensor:
        """
        Codifica el estado de selección para un lote.

        Args:
            disposition: (N, 3) disposición actual.
            opp_initial_warrior: (N,) ID del primer guerrero del oponente.
            opp_initial_position: (N,) posición del primer guerrero del oponente.

        Returns:
            Tensor de forma (N, dim_estado_selección).
        """
        catalog_batch = self._catalog_ids.unsqueeze(0).expand(self.N, -1)
        return ChooseStateV.encode_choose_state_batch(
            disposition,
            catalog_batch,
            opp_initial_warrior,
            opp_initial_position,
        )

    def _replay_turn_and_selection(
        self,
        player: Any,
        selection_states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        selection_actions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        reward_acum: torch.Tensor,
        player_name: str,
        batch_idx: int,
        skip_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Ejecuta el replay de turno (múltiples veces) y de selección.

        Args:
            player: Agente (P1 o P2).
            selection_states: (s1, s2, s3) estados de selección.
            selection_actions: (a1, a2, a3) acciones de selección.
            reward_acum: (N,) recompensa acumulada al final de la partida.
            player_name: "p1" o "p2" (para logging).
            batch_idx: Índice del lote (para logging).
            skip_mask: (N,) bool, True para partidas a excluir (pool).
        """
        # Replay de turno (múltiples veces)
        loss_turn = None
        for _ in range(constants.TURN_REPLAYS_PER_BATCH):
            loss_turn = player.replay_turn()

        if self.logger and loss_turn is not None:
            self.logger.log_loss(
                batch_idx,
                player.replayed_turn,
                player_name,
                "turn",
                loss_turn,
            )

        # Replay de selección
        self._remember_and_replay_selection_batch(
            selection_states,
            selection_actions,
            reward_acum,
            player,
            player_name,
            batch_idx,
            skip_mask,
        )

    def _remember_and_replay_selection_batch(
        self,
        selection_states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        selection_actions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        reward_acum: torch.Tensor,
        player: Any,
        player_name: str,
        batch_idx: int,
        skip_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Almacena las experiencias de selección y ejecuta el replay.

        Las experiencias se almacenan como transiciones (s1, a1, reward, s2, done=False),
        (s2, a2, reward, s3, done=False), (s3, a3, reward, None, done=True).
        """
        s1, s2, s3 = selection_states
        a1, a2, a3 = selection_actions

        if skip_mask is not None:
            valid = ~skip_mask
            s1, s2, s3 = s1[valid], s2[valid], s3[valid]
            a1, a2, a3 = a1[valid], a2[valid], a3[valid]
            reward_acum = reward_acum[valid]

        if s1.shape[0] > 0:
            rewards = reward_acum
            player.remember_selection_batch(s1, a1, rewards, s2, torch.zeros(s1.shape[0], dtype=torch.bool))
            player.remember_selection_batch(s2, a2, rewards, s3, torch.zeros(s2.shape[0], dtype=torch.bool))
            player.remember_selection_batch(s3, a3, rewards, None, torch.ones(s3.shape[0], dtype=torch.bool))

        for _ in range(constants.SELECTION_REPLAYS_PER_BATCH):
            loss = player.replay_selection()
            if self.logger and loss is not None:
                self.logger.log_loss(
                    batch_idx,
                    player.replayed_selection,
                    player_name,
                    "selection",
                    loss,
                )

    def _remember_turn_batch(
        self,
        player: Any,
        experience: Any,
        skip_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Almacena una experiencia de turno en el buffer del jugador, aplicando skip_mask si es necesario.
        """
        if skip_mask is not None:
            valid = ~skip_mask
            states = experience.states[valid]
            actions = experience.actions[valid]
            rewards = experience.rewards[valid]
            next_states = experience.next_states[valid]
            dones = experience.dones[valid]
            alive = experience.alive[valid]
            types = experience.types[valid]
            cooldowns = experience.cooldowns[valid]
            opp_types = experience.opp_types[valid]
            next_types = experience.next_types[valid]
            next_alive = experience.next_alive[valid]
            next_cooldowns = experience.next_cooldowns[valid]
            next_opp_types = experience.next_opp_types[valid]
        else:
            states = experience.states
            actions = experience.actions
            rewards = experience.rewards
            next_states = experience.next_states
            dones = experience.dones
            alive = experience.alive
            types = experience.types
            cooldowns = experience.cooldowns
            opp_types = experience.opp_types
            next_types = experience.next_types
            next_alive = experience.next_alive
            next_cooldowns = experience.next_cooldowns
            next_opp_types = experience.next_opp_types

        if states.shape[0] == 0:
            return

        player.remember_turn_batch(
            states,
            actions,
            rewards,
            next_states,
            dones,
            alive,
            types,
            cooldowns,
            opp_types,
            next_types,
            next_alive,
            next_cooldowns,
            next_opp_types,
        )

    def _build_observations(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construye las observaciones normalizadas para P1 y P2.
        """
        speed_p1 = self.environment.speed_por_tipo[self.environment.p1_disposition] / 20.0
        speed_p2 = self.environment.speed_por_tipo[self.environment.p2_disposition] / 20.0

        maxh_p1 = self.environment.max_health_por_tipo[self.environment.p1_disposition]
        maxh_p2 = self.environment.max_health_por_tipo[self.environment.p2_disposition]

        health_norm_p1 = self.environment.p1_healths / maxh_p1
        health_norm_p2 = self.environment.p2_healths / maxh_p2

        life_p1 = torch.where(
            self.environment.p1_alive,
            health_norm_p1,
            torch.zeros_like(health_norm_p1),
        )
        life_p2 = torch.where(
            self.environment.p2_alive,
            health_norm_p2,
            torch.zeros_like(health_norm_p2),
        )

        turn_norm = (self.environment.turn_number.float() / constants.MAX_TURNS).clamp(max=1.0)

        obs1 = ObservationV.normalize_batch(
            self.environment.p1_disposition,
            self.environment.p1_alive,
            speed_p1,
            health_norm_p1,
            self.environment.p1_cooldowns,
            life_p2,
            self.environment.p2_disposition,
            turn_norm,
        )
        obs2 = ObservationV.normalize_batch(
            self.environment.p2_disposition,
            self.environment.p2_alive,
            speed_p2,
            health_norm_p2,
            self.environment.p2_cooldowns,
            life_p1,
            self.environment.p1_disposition,
            turn_norm,
        )

        return obs1, obs2

    def _turn_mixed_opponent(
        self,
        obs2_tensor: torch.Tensor,
        from_pool: torch.Tensor,
        grouped_opponents: Dict[int, Tuple[Any, torch.Tensor]],
        p2_training_player: Any,
    ) -> torch.Tensor:
        """
        Obtiene acciones para P2, combinando el jugador entrenable y los oponentes de la pool.
        """
        # Acciones base del jugador entrenable (para todas las partidas)
        actions = p2_training_player.turn(
            obs2_tensor,
            self.environment.p2_disposition,
            self.environment.p2_cooldowns,
            self.environment.p2_alive,
            self.environment.p1_disposition,
        )

        # Sobrescribir acciones para las partidas asignadas a la pool
        for cp_id, (opponent, indices) in grouped_opponents.items():
            pool_actions = opponent.turn(
                obs2_tensor,
                self.environment.p2_disposition,
                self.environment.p2_cooldowns,
                self.environment.p2_alive,
                self.environment.p1_disposition,
            )
            actions[indices] = pool_actions[indices]

        return actions

    def _set_epsilons(self, epsilon_turn: float, epsilon_sel: Optional[float]) -> Dict[str, float]:
        """Guarda los epsilons actuales y establece los nuevos."""
        backup = {}
        for name, player in (("p1", self.player1), ("p2", self.player2)):
            if hasattr(player, "epsilon_turn"):
                backup[f"{name}_turn"] = player.epsilon_turn
                player.epsilon_turn = epsilon_turn
            if epsilon_sel is not None and hasattr(player, "epsilon_sel"):
                backup[f"{name}_sel"] = player.epsilon_sel
                player.epsilon_sel = epsilon_sel
        return backup

    def _restore_epsilons(self, backup: Dict[str, float]) -> None:
        """Restaura los epsilons guardados."""
        for name, player in (("p1", self.player1), ("p2", self.player2)):
            if f"{name}_turn" in backup:
                player.epsilon_turn = backup[f"{name}_turn"]
            if f"{name}_sel" in backup:
                player.epsilon_sel = backup[f"{name}_sel"]

    def _load_if_exists(self) -> None:
        """Carga los modelos de P1 y P2 si existen los archivos."""
        if hasattr(self.player1, "load_model") and os.path.exists(self.pathp1_1) and os.path.exists(self.pathp1_2):
            self.player1.load_model(self.pathp1_1, self.pathp1_2)
        if hasattr(self.player2, "load_model") and os.path.exists(self.pathp2_1) and os.path.exists(self.pathp2_2):
            self.player2.load_model(self.pathp2_1, self.pathp2_2)

    @staticmethod
    def _save_if_supported(player: Any, path1: str, path2: str) -> None:
        """Guarda el modelo si el jugador tiene método save_model."""
        if hasattr(player, "save_model"):
            player.save_model(path1, path2)

    def _print_progress(self, episode: int, total_episodes: int, start_time: float) -> None:
        """Muestra el progreso del entrenamiento."""
        if total_episodes == 0:
            return
        if self.progress_every and episode % self.progress_every != 0 and episode != total_episodes - 1:
            return

        elapsed = time.time() - start_time
        pct = (episode + 1) / total_episodes * 100
        eps_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0
        remaining = total_episodes - episode - 1
        eta = remaining / eps_per_sec if eps_per_sec > 0 else 0

        print(
            f"\r[{pct:5.1f}%] Lote {episode + 1}/{total_episodes} "
            f"| {eps_per_sec:6.1f} lotes/s | ETA {self._format_time(eta)}   ",
            end="",
            flush=True,
        )

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Formatea segundos a una cadena legible (h:m:s)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        if hours > 0:
            return f"{hours}h {minutes}m {secs:.0f}s"
        if minutes > 0:
            return f"{minutes}m {secs:.0f}s"
        return f"{secs:.1f}s"