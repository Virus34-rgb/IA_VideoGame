"""
Entrenador para Castle Game.
"""
import os
import time
from typing import Optional, Any, Tuple, Dict

import torch
from AI.Meta.castle_v import CastleV
from AI.Meta.shop_heuristics import decidir_compra_batch
import constants

from AI.Agent.choose_state import ChooseStateV
from AI.Agent.nstep_buffer import NStepBuffer
from AI.Agent.observationV import ObservationV
from AI.Agent.eloRating import EloRating
from AI.Environment.abilitySampling import sample_abilities_batch_all_types


class TrainerV:
    def __init__(
        self, player1, player2, environment, opponent_pool,
        train_batches, eval_batches, pathp1_1, pathp1_2, pathp2_1, pathp2_2,
        path_stats, path_stats2, logger=None, snapshot_every=1000, progress_every=1,
    ) -> None:
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

        self._catalog_ids = torch.arange(1, constants.WARRIOR_QUANTITY + 1, dtype=torch.long)

        self._opponent_from_pool_mask = torch.zeros(self.N, dtype=torch.bool)
        self._grouped_opponents: Dict[int, Tuple[Any, torch.Tensor]] = {}
        self._current_catalog_abilities: torch.Tensor = torch.zeros(
            self.N, constants.WARRIOR_QUANTITY, constants.ABILITIES_PER_WARRIOR, dtype=torch.long,
        )

        # NUEVO: se crean siempre (coste despreciable); solo se usan si USE_META_GAME=True
        self.p1_castle = CastleV(self.N)
        self.p2_castle = CastleV(self.N)

    def train(self) -> None:
        self._load_if_exists()
        self._run(
            batches=self.train_batches, epsilon_turn=0.5, epsilon_sel=None,
            learn_p1=True, learn_p2=True, stats_path=self.path_stats, restore_epsilon=True,
        )
        self._save_if_supported(self.player1, self.pathp1_1, self.pathp1_2)
        self._save_if_supported(self.player2, self.pathp2_1, self.pathp2_2)

    def evaluate(self) -> None:
        self.environment.stats.reset()
        self._load_if_exists()
        self.player1.selection_network.eval()
        self.player1.turn_network.eval()
        self.player2.selection_network.eval()
        self.player2.turn_network.eval()
        self._run(
            batches=self.eval_batches, epsilon_turn=0.02, epsilon_sel=0.02,
            learn_p1=False, learn_p2=False, stats_path=self.path_stats2, restore_epsilon=True,
        )
        self.player1.selection_network.train()
        self.player1.turn_network.train()
        self.player2.selection_network.train()
        self.player2.turn_network.train()

    def _run(self, batches, epsilon_turn, epsilon_sel, learn_p1, learn_p2, stats_path, restore_epsilon) -> None:
        save_every = max(1, int(batches * constants.SAVE_MODEL_FRACTION))
        pool_every = max(1, int(batches * constants.POOL_RANGE_FRACTION))
        snapshot_every = max(1, batches // 50)

        start_time = time.time()

        p2_training_player = self.player2
        self._opponent_from_pool_mask = torch.zeros(self.N, dtype=torch.bool)
        self._grouped_opponents = {}

        for batch_idx in range(batches):
            if batch_idx != 0 and batch_idx % save_every == 0:
                self.opponent_pool.save_version(p2_training_player)

            if batch_idx != 0 and batch_idx % pool_every == 0:
                from_pool, checkpoint_idx = self.opponent_pool.sample_assignment(self.N, constants.POOL_PORCENTAGE, self.player1.elo)
                self._opponent_from_pool_mask = from_pool
                self._grouped_opponents = (
                    self.opponent_pool.build_grouped_opponents(checkpoint_idx, self.player1.__class__, self.N, self.environment)
                    if from_pool.any() else {}
                )

            self._run_batch(batch_idx, learn_p1, learn_p2, p2_training_player)

            winners = self.environment.winner
            S_p1 = torch.where(winners == 0, torch.ones_like(winners, dtype=torch.float),
                torch.where(winners == 1, torch.zeros_like(winners, dtype=torch.float), torch.full_like(winners, 0.5, dtype=torch.float)))
            S_p2 = torch.where(winners == 1, torch.ones_like(winners, dtype=torch.float),
                torch.where(winners == 0, torch.zeros_like(winners, dtype=torch.float), torch.full_like(winners, 0.5, dtype=torch.float)))
            mask_no_pool = ~self._opponent_from_pool_mask
            n = mask_no_pool.sum()
            if learn_p1 or learn_p2:
                if n != 0:
                    S_agg_p1 = S_p1[mask_no_pool].mean().item()
                    elo1 = self.player1.elo
                    elo2 = self.player2.elo
                    expected = EloRating.expected_score(elo1, elo2)
                    self.player1.elo = EloRating.update_elo(elo1, expected, S_agg_p1)
                    self.player2.elo = EloRating.update_elo(elo2, 1 - expected, 1 - S_agg_p1)
                if self._opponent_from_pool_mask.sum() != 0:
                    for cp_id, (jugador, partida_indices) in self._grouped_opponents.items():
                        S_agg_cp = S_p2[partida_indices].mean().item()
                        elo1 = self.player1.elo
                        elo2 = self.opponent_pool.get_elo(cp_id)
                        expected = EloRating.expected_score(elo1, elo2)
                        self.player1.elo = EloRating.update_elo(elo1, expected, 1 - S_agg_cp)
                        new_elo2 = EloRating.update_elo(elo2, 1 - expected, S_agg_cp)
                        self.opponent_pool.update_elo(cp_id, new_elo2)

            if self.logger and (learn_p1 or learn_p2) and snapshot_every and batch_idx % snapshot_every == 0:
                self.logger.log_snapshot(
                    batch_idx, self.player1, p2_training_player, self.environment.stats,
                    elo_p1=self.player1.elo, elo_p2=p2_training_player.elo, pool_elos=self.opponent_pool.elos,
                )

            self._print_progress(batch_idx, batches, start_time)

        if batches > 0:
            print()

        self.environment.stats.guardar_stats(
            stats_path, self.environment.warriors_classes,
            p1_elo=self.player1.elo, p2_elo=p2_training_player.elo, pool_elos=self.opponent_pool.elos,
        )

    def _run_batch(self, batch_idx: int, learn_p1: bool, learn_p2: bool, p2_training_player) -> None:
        if not constants.USE_META_GAME:
            # CAMBIADO: el sorteo de catálogo solo hace falta en modo sin meta-juego
            self._current_catalog_abilities = sample_abilities_batch_all_types(
                self.environment.warriors_classes, self.N, constants.ABILITIES_PER_WARRIOR
            )

        self.environment.reset()
        self.player1.reset_noise()
        self.player2.reset_noise()
        selection_states_p1, selection_actions_p1, selection_states_p2, selection_actions_p2 = self._select_teams(p2_training_player)

        obs1_tensor, obs2_tensor = self._build_observations()
        reward1_acum = torch.zeros(self.N)
        reward2_acum = torch.zeros(self.N)

        n_steps_buffer_p1 = NStepBuffer(n_step=constants.N_STEP, gamma=constants.DISCOUNT_FACTOR)
        n_steps_buffer_p2 = NStepBuffer(n_step=constants.N_STEP, gamma=constants.DISCOUNT_FACTOR)

        while not self.environment.ended.all():
            self._run_turn(obs1_tensor, obs2_tensor, n_steps_buffer_p1, n_steps_buffer_p2, p2_training_player, learn_p1, learn_p2)
            obs1_tensor, obs2_tensor = self._build_observations()
            reward1_acum += self._last_reward1
            reward2_acum += self._last_reward2

        if constants.USE_META_GAME:
            # CORREGIDO: antes se pasaba p1_castle_slots dos veces
            self._run_meta_step(
                self.environment.p1_castle_slots, self.environment.p2_castle_slots,
                self.environment.p1_alive, self.environment.p2_alive,
            )

        if learn_p1:
            for experience in n_steps_buffer_p1.flush():
                self._remember_turn_batch(self.player1, experience)
        if learn_p2:
            for experience in n_steps_buffer_p2.flush():
                self._remember_turn_batch(p2_training_player, experience, skip_mask=self._opponent_from_pool_mask)

        if learn_p1:
            self._replay_turn_and_selection(self.player1, selection_states_p1, selection_actions_p1, reward1_acum, "p1", batch_idx)
            if self.train_batches != 0:
                self.player1.update_beta()

        if learn_p2:
            self._replay_turn_and_selection(
                p2_training_player, selection_states_p2, selection_actions_p2, reward2_acum, "p2", batch_idx,
                skip_mask=self._opponent_from_pool_mask,
            )
            if self.train_batches != 0:
                p2_training_player.update_beta()

        self.environment.stats.total_reward_p1 += reward1_acum.sum().item()
        self.environment.stats.total_reward_p2 += reward2_acum.sum().item()

    def _run_turn(self, obs1_tensor, obs2_tensor, n_steps_buffer_p1, n_steps_buffer_p2, p2_training_player, learn_p1, learn_p2) -> None:
        p1_alive_now = self.environment.p1_alive
        p2_alive_now = self.environment.p2_alive
        p1_types_now = self.environment.p1_disposition
        p1_cd_now = self.environment.p1_cooldowns
        p2_types_now = self.environment.p2_disposition
        p2_cd_now = self.environment.p2_cooldowns
        p1_opp_types_now = self.environment.p2_disposition
        p2_opp_types_now = self.environment.p1_disposition
        p1_abilities_now = self.environment.p1_instance_abilities
        p2_abilities_now = self.environment.p2_instance_abilities

        action_p1 = self.player1.turn(
            obs1_tensor, self.environment.p1_disposition, self.environment.p1_cooldowns,
            self.environment.p1_alive, self.environment.p2_disposition,
            self.environment.p1_instance_abilities,
        )
        action_p2 = self._turn_mixed_opponent(obs2_tensor, self._opponent_from_pool_mask, self._grouped_opponents, p2_training_player)

        state, reward1, reward2, ended = self.environment.turn(action_p1, action_p2)

        self._last_reward1 = reward1
        self._last_reward2 = reward2

        if learn_p1:
            exp_p1 = n_steps_buffer_p1.push(
                obs1_tensor, action_p1, reward1, ended, p1_alive_now, p1_types_now, p1_cd_now, p1_opp_types_now,
                p1_abilities_now,
            )
            if exp_p1 is not None:
                self._remember_turn_batch(self.player1, exp_p1)

        if learn_p2:
            exp_p2 = n_steps_buffer_p2.push(
                obs2_tensor, action_p2, reward2, ended, p2_alive_now, p2_types_now, p2_cd_now, p2_opp_types_now,
                p2_abilities_now,
            )
            if exp_p2 is not None:
                self._remember_turn_batch(p2_training_player, exp_p2, skip_mask=self._opponent_from_pool_mask)

    def _run_meta_step(self, castle_slots_p1, castle_slots_p2, p1_alive_final, p2_alive_final):
        self.p1_castle.envejecer_heroes(castle_slots_p1)
        self.p2_castle.envejecer_heroes(castle_slots_p2)
        self.p1_castle.resolver_muertes(self._traducir_muertes_combate(castle_slots_p1, p1_alive_final))
        self.p2_castle.resolver_muertes(self._traducir_muertes_combate(castle_slots_p2, p2_alive_final))
        self.p1_castle.gold += constants.GOLD_POR_BATALLA
        self.p2_castle.gold += constants.GOLD_POR_BATALLA

        warrior_most_use_p1, warrior_most_use_p2 = self._tipo_mas_repetido()
        for _ in range(constants.MAX_DEATHS_PER_TEAM):
            mask_compra_p1, tipo_p1 = decidir_compra_batch(self.p1_castle, warrior_most_use_p1)
            mask_compra_p2, tipo_p2 = decidir_compra_batch(self.p2_castle, warrior_most_use_p2)
            self.p1_castle.comprar_heroes(mask_compra_p1, tipo_p1)
            self.p2_castle.comprar_heroes(mask_compra_p2, tipo_p2)

    def _tipo_mas_repetido(self):
        # CORREGIDO: self.stats -> self.environment.stats (TrainerV no tiene atributo stats propio)
        tipo_p1 = torch.argmax(self.environment.stats._p1_warrior_use_tensor).item() + 1
        tipo_p2 = torch.argmax(self.environment.stats._p2_warrior_use_tensor).item() + 1
        return torch.full((self.N,), tipo_p1, dtype=torch.long), torch.full((self.N,), tipo_p2, dtype=torch.long)

    def _traducir_muertes_combate(self, castle_slots, alive_final):
        N = castle_slots.shape[0]
        max_size = constants.MAX_CASTLE_SIZE
        mask_muertes = torch.zeros((N, max_size), dtype=torch.bool)
        for slot in range(3):
            muertos = ~alive_final[:, slot]
            ids = castle_slots[muertos, slot]
            mask_muertes[muertos, ids] = True
        return mask_muertes

    def _select_teams(self, p2_training_player):
        if constants.USE_META_GAME:
            return self._select_teams_castle(p2_training_player)
        return self._select_teams_catalog(p2_training_player)

    # ------------------------------------------------------------
    # Draft — modo castillo
    # ------------------------------------------------------------
    def _select_teams_castle(self, p2_training_player):
        indices = torch.arange(self.N)
        used_p1 = torch.zeros(self.N, constants.MAX_CASTLE_SIZE, dtype=torch.bool)
        used_p2 = torch.zeros(self.N, constants.MAX_CASTLE_SIZE, dtype=torch.bool)

        cstate1_1 = self._encode_choose_batch_castle(torch.zeros(self.N, dtype=torch.long), torch.zeros(self.N, dtype=torch.long), self.p1_castle)
        cstate2_1 = self._encode_choose_batch_castle(torch.zeros(self.N, dtype=torch.long), torch.zeros(self.N, dtype=torch.long), self.p2_castle)

        slot1_1, pos1_1, action1_1 = self.player1.selection(cstate1_1, self.environment.p1_disposition, torch.zeros(self.N, dtype=torch.long), self.p1_castle.castle_alive, used_p1)
        slot2_1, pos2_1, action2_1 = p2_training_player.selection(cstate2_1, self.environment.p2_disposition, torch.zeros(self.N, dtype=torch.long), self.p2_castle.castle_alive, used_p2)
        used_p1[indices, slot1_1] = True
        used_p2[indices, slot2_1] = True

        warr1_1_type = self.p1_castle.castle_types[indices, slot1_1]
        warr2_1_type = self.p2_castle.castle_types[indices, slot2_1]
        health1 = self.environment.max_health_por_tipo[warr1_1_type]
        health2 = self.environment.max_health_por_tipo[warr2_1_type]   
        abilities1 = self.p1_castle.castle_abilities[indices, slot1_1]
        abilities2 = self.p2_castle.castle_abilities[indices, slot2_1]  
        self.environment.team_selection(warr1_1_type, pos1_1, warr2_1_type, pos2_1, selected=0, health1=health1, health2=health2, abilities1=abilities1, abilities2=abilities2)

        cstate1_2 = self._encode_choose_batch_castle(warr2_1_type, pos2_1 + 1, self.p1_castle)
        cstate2_2 = self._encode_choose_batch_castle(warr1_1_type, pos1_1 + 1, self.p2_castle)

        slot1_2, pos1_2, action1_2 = self.player1.selection(cstate1_2, self.environment.p1_disposition, warr2_1_type, self.p1_castle.castle_alive, used_p1)
        slot2_2, pos2_2, action2_2 = p2_training_player.selection(cstate2_2, self.environment.p2_disposition, warr1_1_type, self.p2_castle.castle_alive, used_p2)
        used_p1[indices, slot1_2] = True
        used_p2[indices, slot2_2] = True

        warr1_2_type = self.p1_castle.castle_types[indices, slot1_2]
        warr2_2_type = self.p2_castle.castle_types[indices, slot2_2]
        health1 = self.environment.max_health_por_tipo[warr1_2_type]
        health2 = self.environment.max_health_por_tipo[warr2_2_type]
        abilities1 = self.p1_castle.castle_abilities[indices, slot1_2]
        abilities2 = self.p2_castle.castle_abilities[indices, slot2_2]  
        self.environment.team_selection(warr1_2_type, pos1_2, warr2_2_type, pos2_2, selected=1, health1=health1, health2=health2, abilities1=abilities1, abilities2=abilities2)

        cstate1_3 = self._encode_choose_batch_castle(warr2_1_type, pos2_1 + 1, self.p1_castle)
        cstate2_3 = self._encode_choose_batch_castle(warr1_1_type, pos1_1 + 1, self.p2_castle)

        slot1_3, pos1_3, action1_3 = self.player1.selection(cstate1_3, self.environment.p1_disposition, warr2_1_type, self.p1_castle.castle_alive, used_p1)
        slot2_3, pos2_3, action2_3 = p2_training_player.selection(cstate2_3, self.environment.p2_disposition, warr1_1_type, self.p2_castle.castle_alive, used_p2)

        warr1_3_type = self.p1_castle.castle_types[indices, slot1_3]
        warr2_3_type = self.p2_castle.castle_types[indices, slot2_3]
        health1 = self.environment.max_health_por_tipo[warr1_3_type]
        health2 = self.environment.max_health_por_tipo[warr2_3_type]
        abilities1 = self.p1_castle.castle_abilities[indices, slot1_3]
        abilities2 = self.p2_castle.castle_abilities[indices, slot2_3]  

        self.environment.p1_castle_slots[:, 0] = slot1_1
        self.environment.p1_castle_slots[:, 1] = slot1_2
        self.environment.p1_castle_slots[:, 2] = slot1_3
        self.environment.p2_castle_slots[:, 0] = slot2_1
        self.environment.p2_castle_slots[:, 1] = slot2_2
        self.environment.p2_castle_slots[:, 2] = slot2_3

        self.environment.team_selection(warr1_3_type, pos1_3, warr2_3_type, pos2_3, selected=2, health1=health1, health2=health2, abilities1=abilities1, abilities2=abilities2)

        selection_states_p1 = (cstate1_1, cstate1_2, cstate1_3)
        selection_actions_p1 = (action1_1, action1_2, action1_3)
        selection_states_p2 = (cstate2_1, cstate2_2, cstate2_3)
        selection_actions_p2 = (action2_1, action2_2, action2_3)

        return selection_states_p1, selection_actions_p1, selection_states_p2, selection_actions_p2

    def _encode_choose_batch_castle(self, opp_initial_warrior, opp_initial_position, castle):
        return ChooseStateV.encode_choose_state_batch_castle(
            castle.castle_types, castle.castle_abilities, castle.castle_abilities_levels,
            castle.battle_fought, castle.castle_alive, castle.gold,
            opp_initial_warrior, opp_initial_position,
        )

    # ------------------------------------------------------------
    # Draft — modo catálogo (histórico)
    # ------------------------------------------------------------
    def _select_teams_catalog(self, p2_training_player):
        indices = torch.arange(self.N)

        cstate1_1 = self._encode_choose_batch_catalog(self.environment.p1_disposition, torch.zeros(self.N, dtype=torch.long), torch.zeros(self.N, dtype=torch.long))
        cstate2_1 = self._encode_choose_batch_catalog(self.environment.p2_disposition, torch.zeros(self.N, dtype=torch.long), torch.zeros(self.N, dtype=torch.long))

        warr1_1, pos1_1, action1_1 = self.player1.selection(cstate1_1, self.environment.p1_disposition, torch.zeros(self.N, dtype=torch.long))
        warr2_1, pos2_1, action2_1 = p2_training_player.selection(cstate2_1, self.environment.p2_disposition, torch.zeros(self.N, dtype=torch.long))

        health1 = self.environment.max_health_por_tipo[warr1_1]
        health2 = self.environment.max_health_por_tipo[warr2_1]
        abilities1 = self._current_catalog_abilities[indices, warr1_1 - 1]
        abilities2 = self._current_catalog_abilities[indices, warr2_1 - 1]
        self.environment.team_selection(warr1_1, pos1_1, warr2_1, pos2_1, selected=0, health1=health1, health2=health2, abilities1=abilities1, abilities2=abilities2)

        cstate1_2 = self._encode_choose_batch_catalog(self.environment.p1_disposition, warr2_1, pos2_1 + 1)
        cstate2_2 = self._encode_choose_batch_catalog(self.environment.p2_disposition, warr1_1, pos1_1 + 1)

        warr1_2, pos1_2, action1_2 = self.player1.selection(cstate1_2, self.environment.p1_disposition, warr2_1)
        warr2_2, pos2_2, action2_2 = p2_training_player.selection(cstate2_2, self.environment.p2_disposition, warr1_1)

        health1 = self.environment.max_health_por_tipo[warr1_2]
        health2 = self.environment.max_health_por_tipo[warr2_2]
        abilities1 = self._current_catalog_abilities[indices, warr1_2 - 1]
        abilities2 = self._current_catalog_abilities[indices, warr2_2 - 1]
        self.environment.team_selection(warr1_2, pos1_2, warr2_2, pos2_2, selected=1, health1=health1, health2=health2, abilities1=abilities1, abilities2=abilities2)

        cstate1_3 = self._encode_choose_batch_catalog(self.environment.p1_disposition, warr2_1, pos2_1 + 1)
        cstate2_3 = self._encode_choose_batch_catalog(self.environment.p2_disposition, warr1_1, pos1_1 + 1)

        warr1_3, pos1_3, action1_3 = self.player1.selection(cstate1_3, self.environment.p1_disposition, warr2_1)
        warr2_3, pos2_3, action2_3 = p2_training_player.selection(cstate2_3, self.environment.p2_disposition, warr1_1)

        health1 = self.environment.max_health_por_tipo[warr1_3]
        health2 = self.environment.max_health_por_tipo[warr2_3]
        abilities1 = self._current_catalog_abilities[indices, warr1_3 - 1]
        abilities2 = self._current_catalog_abilities[indices, warr2_3 - 1]
        self.environment.team_selection(warr1_3, pos1_3, warr2_3, pos2_3, selected=2, health1=health1, health2=health2, abilities1=abilities1, abilities2=abilities2)

        selection_states_p1 = (cstate1_1, cstate1_2, cstate1_3)
        selection_actions_p1 = (action1_1, action1_2, action1_3)
        selection_states_p2 = (cstate2_1, cstate2_2, cstate2_3)
        selection_actions_p2 = (action2_1, action2_2, action2_3)

        return selection_states_p1, selection_actions_p1, selection_states_p2, selection_actions_p2

    def _encode_choose_batch_catalog(self, disposition, opp_initial_warrior, opp_initial_position):
        catalog_batch = self._catalog_ids.unsqueeze(0).expand(self.N, -1)
        return ChooseStateV.encode_choose_state_batch_catalog(
            disposition, catalog_batch, opp_initial_warrior, opp_initial_position,
            self._current_catalog_abilities,
        )

    def _replay_turn_and_selection(self, player, selection_states, selection_actions, reward_acum, player_name, batch_idx, skip_mask=None) -> None:
        loss_turn = None
        for _ in range(constants.TURN_REPLAYS_PER_BATCH):
            loss_turn = player.replay_turn()

        if self.logger and loss_turn is not None:
            self.logger.log_loss(batch_idx, player.replayed_turn, player_name, "turn", loss_turn)

        self._remember_and_replay_selection_batch(selection_states, selection_actions, reward_acum, player, player_name, batch_idx, skip_mask)

    def _remember_and_replay_selection_batch(self, selection_states, selection_actions, reward_acum, player, player_name, batch_idx, skip_mask=None) -> None:
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
                self.logger.log_loss(batch_idx, player.replayed_selection, player_name, "selection", loss)

    def _remember_turn_batch(self, player, experience, skip_mask=None) -> None:
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
            instance_abilities = experience.instance_abilities[valid]
            next_instance_abilities = experience.next_instance_abilities[valid]
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
            instance_abilities = experience.instance_abilities
            next_instance_abilities = experience.next_instance_abilities

        if states.shape[0] == 0:
            return

        player.remember_turn_batch(
            states, actions, rewards, next_states, dones,
            alive, types, cooldowns, opp_types, next_types, next_alive, next_cooldowns, next_opp_types,
            instance_abilities, next_instance_abilities,
        )

    def _build_observations(self):
        speed_p1 = self.environment.speed_por_tipo[self.environment.p1_disposition] / 20.0
        speed_p2 = self.environment.speed_por_tipo[self.environment.p2_disposition] / 20.0

        maxh_p1 = self.environment.max_health_por_tipo[self.environment.p1_disposition]
        maxh_p2 = self.environment.max_health_por_tipo[self.environment.p2_disposition]

        health_norm_p1 = self.environment.p1_healths / maxh_p1
        health_norm_p2 = self.environment.p2_healths / maxh_p2

        life_p1 = torch.where(self.environment.p1_alive, health_norm_p1, torch.zeros_like(health_norm_p1))
        life_p2 = torch.where(self.environment.p2_alive, health_norm_p2, torch.zeros_like(health_norm_p2))

        turn_norm = (self.environment.turn_number.float() / constants.MAX_TURNS).clamp(max=1.0)

        obs1 = ObservationV.normalize_batch(
            self.environment.p1_disposition, self.environment.p1_alive, speed_p1, health_norm_p1,
            self.environment.p1_cooldowns, life_p2, self.environment.p2_disposition, turn_norm,
            self.environment.p1_instance_abilities,
        )
        obs2 = ObservationV.normalize_batch(
            self.environment.p2_disposition, self.environment.p2_alive, speed_p2, health_norm_p2,
            self.environment.p2_cooldowns, life_p1, self.environment.p1_disposition, turn_norm,
            self.environment.p2_instance_abilities,
        )
        return obs1, obs2

    def _turn_mixed_opponent(self, obs2_tensor, from_pool, grouped_opponents, p2_training_player):
        actions = p2_training_player.turn(
            obs2_tensor, self.environment.p2_disposition, self.environment.p2_cooldowns,
            self.environment.p2_alive, self.environment.p1_disposition,
            self.environment.p2_instance_abilities,
        )

        for cp_id, (opponent, indices) in grouped_opponents.items():
            pool_actions = opponent.turn(
                obs2_tensor, self.environment.p2_disposition, self.environment.p2_cooldowns,
                self.environment.p2_alive, self.environment.p1_disposition,
                self.environment.p2_instance_abilities,
            )
            actions[indices] = pool_actions[indices]

        return actions

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

    def _load_if_exists(self) -> None:
        if hasattr(self.player1, "load_model") and os.path.exists(self.pathp1_1) and os.path.exists(self.pathp1_2):
            self.player1.load_model(self.pathp1_1, self.pathp1_2)
        if hasattr(self.player2, "load_model") and os.path.exists(self.pathp2_1) and os.path.exists(self.pathp2_2):
            self.player2.load_model(self.pathp2_1, self.pathp2_2)

    @staticmethod
    def _save_if_supported(player, path1, path2):
        if hasattr(player, "save_model"):
            player.save_model(path1, path2)

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
        print(f"\r[{pct:5.1f}%] Lote {episode + 1}/{total_episodes} | {eps_per_sec:6.1f} lotes/s | ETA {self._format_time(eta)}   ", end="", flush=True)

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