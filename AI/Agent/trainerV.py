import os
import time
import torch

from AI.Agent.choose_stateV import Choose_stateV
from AI.Agent.observationV import ObservationV
from constants import MAX_TURNS, POOL_PORCENTAGE, POOL_RANGE_FRACTION, SAVE_MODEL_FRACTION, SELECTION_REPLAYS_PER_BATCH, TURN_REPLAYS_PER_BATCH, WARRIOR_QUANTITY


class TrainerV:
    def __init__(self, player1, player2, environment, opponent_pool, train_batches, eval_batches,
                 pathp1_1, pathp1_2, pathp2_1, pathp2_2, path_stats, path_stats2,
                 logger=None, snapshot_every=1000, progress_every=1):
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
        self._catalog_ids = torch.arange(1, WARRIOR_QUANTITY + 1, dtype=torch.long)
        # NUEVO: estado de asignación de oponentes del pool, per-partida.
        # Se actualiza cada EPISODES_RANGE_POOL lotes dentro de _run.
        self._opponent_from_pool_mask = torch.zeros(self.N, dtype=torch.bool)
        self._grouped_opponents = {}

    def train(self):
        self._load_if_exists()
        self._run(batches=self.train_batches, epsilon_turn=0.5, epsilon_sel=None,
                   learn_p1=True, learn_p2=True, stats_path=self.path_stats, restore_epsilon=True)
        self._save_if_supported(self.player1, self.pathp1_1, self.pathp1_2)
        self._save_if_supported(self.player2, self.pathp2_1, self.pathp2_2)

    def evaluate(self):
        self.environment.stats.reset()
        self._load_if_exists()
        self._run(batches=self.eval_batches, epsilon_turn=0.02, epsilon_sel=0.02,
                   learn_p1=False, learn_p2=False, stats_path=self.path_stats2, restore_epsilon=True)

    def _run(self, batches, epsilon_turn, epsilon_sel, learn_p1, learn_p2, stats_path, restore_epsilon):
        save_every = max(1, int(batches * SAVE_MODEL_FRACTION))
        pool_every = max(1, int(batches * POOL_RANGE_FRACTION))
        snapshot_every = max(1, batches // 50) 
        backup = self._set_epsilons(epsilon_turn, epsilon_sel)
        start_time = time.time()
        p2_training_player = self.player2
        self._opponent_from_pool_mask = torch.zeros(self.N, dtype=torch.bool)
        self._grouped_opponents = {}

        for batch_idx in range(batches):
            if batch_idx != 0 and batch_idx % save_every == 0:
                self.opponent_pool.save_version(p2_training_player)

            if batch_idx != 0 and batch_idx % pool_every == 0:
                from_pool, checkpoint_idx = self.opponent_pool.sample_assignment(self.N, POOL_PORCENTAGE)
                self._opponent_from_pool_mask = from_pool
                self._grouped_opponents = self.opponent_pool.build_grouped_opponents(
                    checkpoint_idx, self.player1.__class__, self.N, self.environment
                ) if from_pool.any() else {}

            self._run_batch(batch_idx, learn_p1, learn_p2, p2_training_player)

            if self.logger and (learn_p1 or learn_p2) and snapshot_every and batch_idx % snapshot_every == 0:
                    self.logger.log_snapshot(batch_idx, self.player1, p2_training_player, self.environment.stats)
            self._print_progress(batch_idx, batches, start_time)

        if batches > 0:
            print()
        self.environment.stats.guardar_stats(stats_path, self.environment.warriors_classes)
        if restore_epsilon:
            self._restore_epsilons(backup)

    def _run_batch(self, batch_idx, learn_p1, learn_p2, p2_training_player):
        state = self.environment.reset()
        cstates1, actions1, cstates2, actions2 = self._select_teams(state, p2_training_player)

        obs1_tensor, obs2_tensor = self._build_observations()
        reward1_acum = torch.zeros(self.N)
        reward2_acum = torch.zeros(self.N)

        while not self.environment.ended.all():
                # crudos ANTES del turno (estado actual)
            p1_alive_now = self.environment.p1_alive
            p2_alive_now = self.environment.p2_alive
            action_p1 = self.player1.turn(obs1_tensor, self.environment.p1_disposition,
                                           self.environment.p1_cooldowns, self.environment.p1_alive,
                                           self.environment.p2_disposition)
            action_p2 = self._turn_mixed_opponent(obs2_tensor, self._opponent_from_pool_mask,
                                                    self._grouped_opponents, p2_training_player)

            state, reward1, reward2, ended = self.environment.turn(action_p1, action_p2)
            next_obs1_tensor, next_obs2_tensor = self._build_observations()
            p1_types_next, p1_alive_next, p1_cd_next = self.environment.p1_disposition, self.environment.p1_alive, self.environment.p1_cooldowns
            p2_types_next, p2_alive_next, p2_cd_next = self.environment.p2_disposition, self.environment.p2_alive, self.environment.p2_cooldowns

            if learn_p1:
                self._remember_turn_batch(self.player1, obs1_tensor, action_p1, reward1, next_obs1_tensor, ended,
                                   alive=p1_alive_now, next_types=p1_types_next, next_alive=p1_alive_next,
                                   next_cooldowns=p1_cd_next, next_opp_types=p2_types_next)
            if learn_p2:
                self._remember_turn_batch(p2_training_player, obs2_tensor, action_p2, reward2, next_obs2_tensor, ended,
                                        alive=p2_alive_now, next_types=p2_types_next, next_alive=p2_alive_next,
                                        next_cooldowns=p2_cd_next, next_opp_types=p1_types_next,
                                        skip_mask=self._opponent_from_pool_mask)

            obs1_tensor, obs2_tensor = next_obs1_tensor, next_obs2_tensor
            reward1_acum += reward1
            reward2_acum += reward2

        if learn_p1:
            for _ in range(TURN_REPLAYS_PER_BATCH):
                loss1_turn = self.player1.replay_turn()
            if self.logger:
                self.logger.log_loss(batch_idx, self.player1.replayed_turn, "p1", "turn", loss1_turn)
            self._remember_and_replay_selection_batch(cstates1, actions1, reward1_acum, self.player1, "p1", batch_idx)
            if self.train_batches != 0:
                self.player1.update_beta()
                self.player1.update_epsilon(n_games=self.N)

        if learn_p2:
            for _ in range(TURN_REPLAYS_PER_BATCH):
                loss2_turn = p2_training_player.replay_turn()
            if self.logger:
                self.logger.log_loss(batch_idx, p2_training_player.replayed_turn, "p2", "turn", loss2_turn)
            self._remember_and_replay_selection_batch(cstates2, actions2, reward2_acum, p2_training_player, "p2",
                                                        batch_idx, skip_mask=self._opponent_from_pool_mask)
            if self.train_batches != 0:
                p2_training_player.update_beta()
                p2_training_player.update_epsilon(n_games=self.N)

        self.environment.stats.total_reward_p1 += reward1_acum.sum().item()
        self.environment.stats.total_reward_p2 += reward2_acum.sum().item()

    def _turn_mixed_opponent(self, obs2_tensor, from_pool, grouped_opponents, p2_training_player):
        actions = p2_training_player.turn(obs2_tensor, self.environment.p2_disposition,
                                           self.environment.p2_cooldowns, self.environment.p2_alive,
                                           self.environment.p1_disposition)
        for cp_id, (jugador, idx_partidas) in grouped_opponents.items():
            acciones_pool = jugador.turn(obs2_tensor, self.environment.p2_disposition,
                                          self.environment.p2_cooldowns, self.environment.p2_alive,
                                          self.environment.p1_disposition)
            actions[idx_partidas] = acciones_pool[idx_partidas]
        return actions

    def _remember_turn_batch(self,player,states,actions,rewards,next_states,
                             ended,alive,next_types,next_alive,next_cooldowns,next_opp_types,skip_mask=None):
        if skip_mask is not None:
            valid = ~skip_mask
            states = states[valid]
            actions = actions[valid]
            rewards = rewards[valid]
            next_states = next_states[valid]
            ended = ended[valid]
            alive = alive[valid]
            next_types = next_types[valid]
            next_alive = next_alive[valid]
            next_cooldowns = next_cooldowns[valid]
            next_opp_types = next_opp_types[valid]

        if states.shape[0] == 0:
            return

        player.remember_turn_batch(states,actions,rewards,next_states,ended,alive,
                                   next_types,next_alive,next_cooldowns,next_opp_types)

    def _remember_and_replay_selection_batch(self,cstates,actions,reward_acum,
                                             player,player_name,batch_idx,skip_mask=None):
        c1, c2, c3 = cstates
        a1, a2, a3 = actions
        if skip_mask is not None:
            valid = ~skip_mask
            c1 = c1[valid]
            c2 = c2[valid]
            c3 = c3[valid]
            a1 = a1[valid]
            a2 = a2[valid]
            a3 = a3[valid]
            reward_acum = reward_acum[valid]

        if c1.shape[0] > 0:
            rewards = reward_acum
            player.remember_selection_batch(c1,a1,rewards,c2,torch.zeros(c1.shape[0], dtype=torch.bool))
            player.remember_selection_batch(c2,a2,rewards,c3,torch.zeros(c2.shape[0], dtype=torch.bool))
            player.remember_selection_batch(c3,a3,rewards,None,torch.ones(c3.shape[0], dtype=torch.bool))

        for _ in range(SELECTION_REPLAYS_PER_BATCH):
            loss = player.replay_selection()
            if self.logger:
                self.logger.log_loss(batch_idx,player.replayed_selection,player_name,"selection",loss)
                
    def _select_teams(self, state, p2_training_player):
        cstate1_1, cstate2_1 = self.createChooseState(
            state,
            torch.zeros(self.N, dtype=torch.long),
            torch.zeros(self.N, dtype=torch.long),
            torch.zeros(self.N, dtype=torch.long),
            torch.zeros(self.N, dtype=torch.long)
        )
        cstate1_1_t = self._encode_choose_batch(self.environment.p1_disposition,torch.zeros(self.N, dtype=torch.long),
                                                torch.zeros(self.N, dtype=torch.long))

        cstate2_1_t = self._encode_choose_batch(self.environment.p2_disposition,torch.zeros(self.N, dtype=torch.long),
                                                torch.zeros(self.N, dtype=torch.long))
        
        warr1_1, pos1_1, action1_1 = self.player1.selection(cstate1_1_t,self.environment.p1_disposition,
                                                            torch.zeros(self.N, dtype=torch.long))
        warr2_1, pos2_1, action2_1 = p2_training_player.selection(cstate2_1_t,self.environment.p2_disposition,
                                                                  torch.zeros(self.N, dtype=torch.long))
        
        health1 = self.environment.max_health_por_tipo[warr1_1]
        health2 = self.environment.max_health_por_tipo[warr2_1]
        
        state = self.environment.team_selection(warr1_1, pos1_1, warr2_1, pos2_1,selected=0, health1=health1
                                                ,health2=health2)

        cstate1_2, cstate2_2 = self.createChooseState(state,warr2_1,pos2_1 + 1,warr1_1,pos1_1 + 1)
        
        cstate1_2_t = self._encode_choose_batch(self.environment.p1_disposition,warr2_1,pos2_1 + 1)
        cstate2_2_t = self._encode_choose_batch(self.environment.p2_disposition,warr1_1,pos1_1 + 1)

        warr1_2, pos1_2, action1_2 = self.player1.selection(cstate1_2_t,self.environment.p1_disposition,warr2_1)
        warr2_2, pos2_2, action2_2 = p2_training_player.selection(cstate2_2_t,self.environment.p2_disposition,warr1_1)

        health1 = self.environment.max_health_por_tipo[warr1_2]
        health2 = self.environment.max_health_por_tipo[warr2_2]

        state = self.environment.team_selection( warr1_2, pos1_2,warr2_2, pos2_2,selected=1,health1=health1,health2=health2)

        cstate1_3, cstate2_3 = self.createChooseState(state,warr2_1,pos2_1 + 1,warr1_1,pos1_1 + 1)
        cstate1_3_t = self._encode_choose_batch(self.environment.p1_disposition,warr2_1,pos2_1 + 1)
        cstate2_3_t = self._encode_choose_batch(self.environment.p2_disposition,warr1_1,pos1_1 + 1)
        
        warr1_3, pos1_3, action1_3 = self.player1.selection(cstate1_3_t,self.environment.p1_disposition,warr2_1)

        warr2_3, pos2_3, action2_3 = p2_training_player.selection(cstate2_3_t,self.environment.p2_disposition,warr1_1)

        health1 = self.environment.max_health_por_tipo[warr1_3]
        health2 = self.environment.max_health_por_tipo[warr2_3]

        self.environment.team_selection(warr1_3, pos1_3,warr2_3, pos2_3,selected=2,health1=health1,health2=health2)

        cstates1 = (cstate1_1, cstate1_2, cstate1_3)
        actions1 = (action1_1, action1_2, action1_3)

        cstates2 = (cstate2_1, cstate2_2, cstate2_3)
        actions2 = (action2_1, action2_2, action2_3)

        return cstates1, actions1, cstates2, actions2

    def createChooseState(self, state, p1_first_warrior, p1_first_pos, p2_first_warrior, p2_first_pos):
        catalogo_batch = self._catalog_ids.unsqueeze(0).expand(self.N, -1)
        cstates1 = Choose_stateV.encode_choose_state_batch(
            self.environment.p1_disposition,
            catalogo_batch,
            p2_first_warrior,
            p2_first_pos
        )
        cstates2 = Choose_stateV.encode_choose_state_batch(
            self.environment.p2_disposition,
            catalogo_batch,
            p1_first_warrior,
            p1_first_pos
        )
        return cstates1, cstates2

    def _encode_choose_batch(self, pl_disposition, opp_initial_warrior, opp_initial_position):
        catalogo_batch = self._catalog_ids.unsqueeze(0).expand(self.N, -1)
        return Choose_stateV.encode_choose_state_batch(
            pl_disposition, catalogo_batch, opp_initial_warrior, opp_initial_position
        )

    def _build_observations(self):
        speed_p1 = self.environment.speed_por_tipo[self.environment.p1_disposition] / 20
        speed_p2 = self.environment.speed_por_tipo[self.environment.p2_disposition] / 20
        maxh_p1 = self.environment.max_health_por_tipo[self.environment.p1_disposition]
        maxh_p2 = self.environment.max_health_por_tipo[self.environment.p2_disposition]
        health_norm_p1 = self.environment.p1_healths / maxh_p1
        health_norm_p2 = self.environment.p2_healths / maxh_p2
        life_p1 = torch.where(self.environment.p1_alive, health_norm_p1, torch.zeros_like(health_norm_p1))
        life_p2 = torch.where(self.environment.p2_alive, health_norm_p2, torch.zeros_like(health_norm_p2))
        turn_norm = (self.environment.turn_number.float() / MAX_TURNS).clamp(max=1.0)

        obs1_tensor = ObservationV.normalize_batch(
            self.environment.p1_disposition, self.environment.p1_alive, speed_p1, health_norm_p1,
            self.environment.p1_cooldowns, life_p2, self.environment.p2_disposition, turn_norm
        )
        obs2_tensor = ObservationV.normalize_batch(
            self.environment.p2_disposition, self.environment.p2_alive, speed_p2, health_norm_p2,
            self.environment.p2_cooldowns, life_p1, self.environment.p1_disposition, turn_norm
        )
        return obs1_tensor, obs2_tensor

    def _load_if_exists(self):
        if hasattr(self.player1, "load_model") and os.path.exists(self.pathp1_1) and os.path.exists(self.pathp1_2):
            self.player1.load_model(self.pathp1_1, self.pathp1_2)
        if hasattr(self.player2, "load_model") and os.path.exists(self.pathp2_1) and os.path.exists(self.pathp2_2):
            self.player2.load_model(self.pathp2_1, self.pathp2_2)

    @staticmethod
    def _save_if_supported(player, path1, path2):
        if hasattr(player, "save_model"):
            player.save_model(path1, path2)

    def _set_epsilons(self, epsilon_turn, epsilon_sel):
        backup = {}
        for name, player in (("p1", self.player1), ("p2", self.player2)):
            if hasattr(player, "epsilon_turn"):
                backup[f"{name}_turn"] = player.epsilon_turn
                player.epsilon_turn = epsilon_turn
            if epsilon_sel is not None and hasattr(player, "epsilon_sel"):
                backup[f"{name}_sel"] = player.epsilon_sel
                player.epsilon_sel = epsilon_sel
        return backup

    def _restore_epsilons(self, backup):
        for name, player in (("p1", self.player1), ("p2", self.player2)):
            if f"{name}_turn" in backup:
                player.epsilon_turn = backup[f"{name}_turn"]
            if f"{name}_sel" in backup:
                player.epsilon_sel = backup[f"{name}_sel"]

    # CORREGIDO: el cuerpo de esta función estaba incompleto (cortado tras
    # `elapsed = ...`) — faltaban pct, eps_per_sec, eta y el print.
    def _print_progress(self, episode, total_episodes, start_time):
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
            end="", flush=True
        )

    @staticmethod
    def _format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        if hours > 0:
            return f"{hours}h {minutes}m {secs:.0f}s"
        if minutes > 0:
            return f"{minutes}m {secs:.0f}s"
        return f"{secs:.1f}s"