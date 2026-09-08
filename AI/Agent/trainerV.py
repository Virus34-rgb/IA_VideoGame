"""
Entrenador para Castle Game.
"""
import os
import time
from typing import Optional, Any, Tuple, Dict

import torch
import wandb
from AI.Environment.abilityData import EffectType
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
        self, player1, player2,playerRusher, environment, opponent_pool,
        train_batches, eval_batches, pathp1_1, pathp1_2, pathp2_1, pathp2_2,
        path_stats, path_stats2, logger=None, snapshot_every=1000, progress_every=1,
    ) -> None:
        self.player1 = player1
        self.player2 = player2
        self.playerRusher = playerRusher
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

        self._opponent_from_pool_mask = torch.zeros(self.N, dtype=torch.bool)
        self._opponent_rusher_mask = torch.zeros(self.N, dtype=torch.bool)
        
        self._grouped_opponents: Dict[int, Tuple[Any, torch.Tensor]] = {}

        self.p1_castle = CastleV(self.N)
        self.p2_castle = CastleV(self.N)

        self._p1_profile = torch.full((self.N, 4), 0.5)
        self._p2_profile = torch.full((self.N, 4), 0.5)

    def train(self) -> None:
        self._load_if_exists()
        self._run(
            batches=self.train_batches, epsilon_turn=0.5, epsilon_sel=None,
            learn_p1=True, learn_p2=True, stats_path=self.path_stats, restore_epsilon=True,
        )
        self._save_if_supported(self.player1, self.pathp1_1, self.pathp1_2)
        self._save_if_supported(self.player2, self.pathp2_1, self.pathp2_2)

    def evaluate(self, fixed_rusher_aggression=None, fixed_rusher_aggression_min=None, fixed_rusher_aggression_max=None) -> None:
        self.environment.stats.reset()
        self._load_if_exists()
        self._set_eval_mode(self.player1)
        self._set_eval_mode(self.player2)
        self._run(
            batches=self.eval_batches, epsilon_turn=0.02, epsilon_sel=0.02,
            learn_p1=False, learn_p2=False, stats_path=self.path_stats2, restore_epsilon=True,
            fixed_rusher_aggression=fixed_rusher_aggression,
            fixed_rusher_aggression_min=fixed_rusher_aggression_min,
            fixed_rusher_aggression_max=fixed_rusher_aggression_max,
        )
        self._set_train_mode(self.player1)
        self._set_train_mode(self.player2)

    @staticmethod
    def _set_eval_mode(player) -> None:
        if hasattr(player, "selection_network"):
            player.selection_network.eval()
        if hasattr(player, "turn_network"):
            player.turn_network.eval()

    @staticmethod
    def _set_train_mode(player) -> None:
        if hasattr(player, "selection_network"):
            player.selection_network.train()
        if hasattr(player, "turn_network"):
            player.turn_network.train()

    def _run(self, batches, epsilon_turn, epsilon_sel, learn_p1, learn_p2, stats_path, restore_epsilon
             , fixed_rusher_aggression=None, fixed_rusher_aggression_min = None,fixed_rusher_aggression_max = None) -> None:
        save_every = max(1, int(batches * constants.SAVE_MODEL_FRACTION))
        pool_every = max(1, int(batches * constants.POOL_RANGE_FRACTION))
        snapshot_every = max(1, batches // 50)

        start_time = time.time()

        p2_training_player = self.player2
        is_rusher_step = p2_training_player is self.playerRusher
        if is_rusher_step:
            self.environment.stats.partidas_vs_rusher = 0
            self.environment.stats.p1_victories_vs_rusher = 0
            self.environment.stats.p2_victories_vs_rusher = 0
            self.environment.stats.empates_vs_rusher = 0
            if fixed_rusher_aggression is not None:
                self.playerRusher.set_aggression(fixed_rusher_aggression)
        
        self._opponent_from_pool_mask = torch.zeros(self.N, dtype=torch.bool)
        self._opponent_rusher_mask = torch.zeros(self.N,dtype=torch.bool)
        self._grouped_opponents = {}

        for batch_idx in range(batches):
            if (learn_p1 or learn_p2) and not is_rusher_step and batch_idx != 0 and batch_idx % save_every == 0:
                self.opponent_pool.save_version(p2_training_player)

            if (learn_p1 or learn_p2) and not is_rusher_step and batch_idx != 0 and batch_idx % pool_every == 0:
                from_pool, checkpoint_idx = self.opponent_pool.sample_assignment(self.N, constants.POOL_PORCENTAGE, self.player1.elo)
                self._opponent_from_pool_mask = from_pool
                self._grouped_opponents = (
                    self.opponent_pool.build_grouped_opponents(checkpoint_idx, self.player1.__class__, self.N, self.environment)
                    if from_pool.any() else {}
                )

            if is_rusher_step:
                self._opponent_rusher_mask = torch.ones(self.N, dtype=torch.bool)
                if fixed_rusher_aggression is None:
                    self.playerRusher.set_aggression(self._sample_rusher_aggression(self.N,fixed_rusher_aggression_min,fixed_rusher_aggression_max))
                    
            elif (learn_p1 or learn_p2) and batch_idx:
                rand = torch.rand(self.N)
                self._opponent_rusher_mask = (rand < constants.RUSHER_OPPONENT_PERCENTAGE) & ~self._opponent_from_pool_mask
                if self._opponent_rusher_mask.any():
                    self.playerRusher.set_aggression(self._sample_rusher_aggression(self.N))

            self._run_batch(batch_idx, learn_p1, learn_p2, p2_training_player, batches)
            
            if learn_p1 and hasattr(self.player1, "update_epsilon"):
                self.player1.update_epsilon(n_games=self.N)
            if learn_p2 and hasattr(p2_training_player, "update_epsilon"):
                p2_training_player.update_epsilon(n_games=self.N)

            self.environment.stats.accumulate_rusher_stats(self.environment.winner, self._opponent_rusher_mask)

            winners = self.environment.winner
            S_p1 = torch.where(winners == 0, torch.ones_like(winners, dtype=torch.float),
                torch.where(winners == 1, torch.zeros_like(winners, dtype=torch.float), torch.full_like(winners, 0.5, dtype=torch.float)))
            S_p2 = torch.where(winners == 1, torch.ones_like(winners, dtype=torch.float),
                torch.where(winners == 0, torch.zeros_like(winners, dtype=torch.float), torch.full_like(winners, 0.5, dtype=torch.float)))
            mask_no_pool = ~self._opponent_from_pool_mask
            mask_p2_normal = mask_no_pool & ~self._opponent_rusher_mask
            n = mask_p2_normal.sum()
            
            if learn_p1 or learn_p2:
                if n != 0:
                    S_agg_p1 = S_p1[mask_p2_normal].mean().item()
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
                if constants.USE_WANDB:
                    partidas = max(self.environment.stats.partidas, 1)
                    wandb.log({
                        "winrate/p1": self.environment.stats.p1_victories / partidas * 100,
                        "winrate/p2": self.environment.stats.p2_victories / partidas * 100,
                        "winrate/draw": self.environment.stats.empates / partidas * 100,
                        "turns/avg": self.environment.stats.total_turns / partidas,
                        "reward/p1_avg": self.environment.stats.total_reward_p1 / partidas,
                        "reward/p2_avg": self.environment.stats.total_reward_p2 / partidas,
                        "elo/p1": self.player1.elo,
                        "elo/p2": p2_training_player.elo,
                    }, step=batch_idx)
                    if self.opponent_pool.elos:
                        pool_elos = list(self.opponent_pool.elos.values())
                        wandb.log({
                            "elo/pool_mean": sum(pool_elos) / len(pool_elos),
                            "elo/pool_max": max(pool_elos),
                        }, step=batch_idx)
                        
                sigmas = {"sel_p1": None, "turn_p1": None, "sel_p2": None, "turn_p2": None}
                if hasattr(self.player1, "mean_sigmas"):
                    p1_sigmas = self.player1.mean_sigmas()
                    sigmas["sel_p1"] = p1_sigmas["sel"]
                    sigmas["turn_p1"] = p1_sigmas["turn"]
                if hasattr(p2_training_player, "mean_sigmas"):
                    p2_sigmas = p2_training_player.mean_sigmas()
                    sigmas["sel_p2"] = p2_sigmas["sel"]
                    sigmas["turn_p2"] = p2_sigmas["turn"]
                sigmas = {k: v for k, v in sigmas.items() if v is not None}

                self.logger.log_snapshot(
                    batch_idx, self.player1, p2_training_player, self.environment.stats,
                    elo_p1=self.player1.elo, elo_p2=p2_training_player.elo, pool_elos=self.opponent_pool.elos,
                    sigmas=sigmas,
                    profile_p1=self._p1_profile.mean(dim=0).tolist(),
                    profile_p2=self._p2_profile.mean(dim=0).tolist(),
                )

            self._print_progress(batch_idx, batches, start_time)

        if batches > 0:
            print()

        self.environment.stats.guardar_stats(
            stats_path, self.environment.warriors_classes,
            p1_elo=self.player1.elo, p2_elo=p2_training_player.elo, pool_elos=self.opponent_pool.elos,
        )

    def _run_batch(self, batch_idx: int, learn_p1: bool, learn_p2: bool, p2_training_player, batches: int) -> None:

        self.environment.reset()
        self._p1_profile = torch.full((self.N, 4), 0.5)
        self._p2_profile = torch.full((self.N, 4), 0.5)
        
        self.player1.reset_noise()
        if(self.player2.__class__.__name__ == "PlayerAIV"):
            self.player2.reset_noise()
        selection_states_p1, selection_actions_p1, selection_states_p2, selection_actions_p2 = self._select_teams(p2_training_player)
        
        self._p1_profile[:, 0] = self._estimate_aggression_prior(self.environment.p1_disposition, self.environment.p1_instance_abilities)
        self._p2_profile[:, 0] = self._estimate_aggression_prior(self.environment.p2_disposition, self.environment.p2_instance_abilities)

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
            self._run_meta_step(
                self.environment.p1_castle_slots, self.environment.p2_castle_slots,
                self.environment.p1_alive, self.environment.p2_alive,
            )

        if learn_p1:
            for experience in n_steps_buffer_p1.flush():
                self._remember_turn_batch(self.player1, experience)
        if learn_p2:
            for experience in n_steps_buffer_p2.flush():
                self._remember_turn_batch(p2_training_player, experience, skip_mask=self._opponent_from_pool_mask | self._opponent_rusher_mask)

        if learn_p1:
            self._replay_turn_and_selection(self.player1, selection_states_p1, selection_actions_p1, reward1_acum, "p1", batch_idx)
            if batches != 0:
                self.player1.update_beta()

        if learn_p2:
            self._replay_turn_and_selection(
                p2_training_player, selection_states_p2, selection_actions_p2, reward2_acum, "p2", batch_idx,
                skip_mask=self._opponent_from_pool_mask | self._opponent_rusher_mask,
            )
            if batches != 0:
                p2_training_player.update_beta()
                
        self.environment.stats.total_reward_p1 += reward1_acum.sum().item()
        reward2_valid = reward2_acum[~self._opponent_from_pool_mask & ~self._opponent_rusher_mask]
        self.environment.stats.total_reward_p2 += reward2_valid.sum().item()

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

        p1_action_mask_now = self.player1.compute_action_mask(
            p1_types_now, p1_cd_now, p1_alive_now, p1_opp_types_now, p1_abilities_now,
        )
        p2_action_mask_now = p2_training_player.compute_action_mask(
            p2_types_now, p2_cd_now, p2_alive_now, p2_opp_types_now, p2_abilities_now,
        )

        action_p1 = self.player1.turn(
            obs1_tensor, self.environment.p1_disposition, self.environment.p1_cooldowns,
            self.environment.p1_alive, self.environment.p2_disposition,
            self.environment.p1_instance_abilities,
        )
        action_p2 = self._turn_mixed_opponent(obs2_tensor, self._opponent_from_pool_mask, self._grouped_opponents, p2_training_player)

        state, reward1, reward2, ended = self.environment.turn(action_p1, action_p2)
        
        self._update_profile_accumulators(
            p1_types_now, p1_abilities_now, p1_action_mask_now, p1_alive_now,
            p2_types_now, p2_abilities_now, p2_action_mask_now, p2_alive_now,
        )

        self._last_reward1 = reward1
        self._last_reward2 = reward2

        if learn_p1:
            exp_p1 = n_steps_buffer_p1.push(
                obs1_tensor, action_p1, reward1, ended, p1_alive_now, p1_types_now, p1_cd_now, p1_opp_types_now,
                p1_abilities_now, p1_action_mask_now,
            )
            if exp_p1 is not None:
                self._remember_turn_batch(self.player1, exp_p1)

        if learn_p2:
            exp_p2 = n_steps_buffer_p2.push(
                obs2_tensor, action_p2, reward2, ended, p2_alive_now, p2_types_now, p2_cd_now, p2_opp_types_now,
                p2_abilities_now, p2_action_mask_now,
            )
            if exp_p2 is not None:
                self._remember_turn_batch(p2_training_player, exp_p2, skip_mask=self._opponent_from_pool_mask | self._opponent_rusher_mask)

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
        usage_p1 = self.environment.stats._p1_warrior_use_ema
        usage_p2 = self.environment.stats._p2_warrior_use_ema
        tipo_p1 = self._sample_categorical_shared(usage_p1)
        tipo_p2 = self._sample_categorical_shared(usage_p2)
        return tipo_p1, tipo_p2

    def _sample_categorical_shared(self, usage: torch.Tensor) -> torch.Tensor:
        """
        Muestrea self.N índices (1..WARRIOR_QUANTITY) de UNA distribución categórica
        compartida (softmax(usage / SHOP_TEMPERATURE)), sin usar torch.multinomial.
        Técnica: CDF acumulada + búsqueda binaria vectorizada (searchsorted).
        """
        probs = torch.softmax(usage / constants.SHOP_TEMPERATURE, dim=0)   # (WARRIOR_QUANTITY,)
        cumprobs = torch.cumsum(probs, dim=0)                               # (WARRIOR_QUANTITY,)
        u = torch.rand(self.N)                                              # (N,)
        idx = torch.searchsorted(cumprobs, u).clamp(max=constants.WARRIOR_QUANTITY - 1)
        return idx + 1
    
    def _sample_rusher_aggression(self, N: int,min : float = None,max : float = None) -> torch.Tensor:
        """
        Sustituye el muestreo uniforme torch.rand(N) por un muestreo por bandas
        ponderadas (constants.RUSHER_AGGRESSION_BANDS / RUSHER_AGGRESSION_BAND_WEIGHTS),
        para dar más exposición a las agresividades donde peor rinde la IA (ver
        diagnóstico: ~14.7% winrate en [0.9,1.0], ~24.3% en torno a 0.5).
        Misma técnica de CDF acumulada + búsqueda binaria vectorizada que
        _sample_categorical_shared, para no introducir torch.multinomial.
        """
        if min is not None and max is not None:
            u_within = torch.rand(N)
            return min + u_within * (max - min)
        
        weights = torch.tensor(constants.RUSHER_AGGRESSION_BAND_WEIGHTS, dtype=torch.float)
        cumprobs = torch.cumsum(weights, dim=0)
        u_band = torch.rand(N)
        band_idx = torch.searchsorted(cumprobs, u_band).clamp(max=len(constants.RUSHER_AGGRESSION_BANDS) - 1)

        bands = torch.tensor(constants.RUSHER_AGGRESSION_BANDS, dtype=torch.float)
        lo = bands[band_idx, 0]
        hi = bands[band_idx, 1]

        u_within = torch.rand(N)
        return lo + u_within * (hi - lo)
    
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
        return self._select_teams_castle(p2_training_player)

    def _select_teams_castle(self, p2_training_player):
        indices = torch.arange(self.N)
        used_p1 = torch.zeros(self.N, constants.MAX_CASTLE_SIZE, dtype=torch.bool)
        used_p2 = torch.zeros(self.N, constants.MAX_CASTLE_SIZE, dtype=torch.bool)

        cstate1_1 = self._encode_choose_batch_castle(torch.zeros(self.N, dtype=torch.long), torch.zeros(self.N, dtype=torch.long), self.p1_castle)
        cstate2_1 = self._encode_choose_batch_castle(torch.zeros(self.N, dtype=torch.long), torch.zeros(self.N, dtype=torch.long), self.p2_castle)

        slot1_1, pos1_1, action1_1 = self.player1.selection(cstate1_1, self.environment.p1_disposition, torch.zeros(self.N, dtype=torch.long), self.p1_castle.castle_alive, used_p1,self.p1_castle.castle_types)
        slot2_1, pos2_1, action2_1 = p2_training_player.selection(cstate2_1, self.environment.p2_disposition, torch.zeros(self.N, dtype=torch.long), self.p2_castle.castle_alive, used_p2,self.p2_castle.castle_types)
        slot2_1R, pos2_1R, action2_1R = self.playerRusher.selection(cstate2_1, self.environment.p2_disposition, torch.zeros(self.N, dtype=torch.long), self.p2_castle.castle_alive, used_p2,self.p2_castle.castle_types)
        slot2_1 = torch.where(self._opponent_rusher_mask,slot2_1R,slot2_1)
        pos2_1 = torch.where(self._opponent_rusher_mask,pos2_1R,pos2_1)
        action2_1 = torch.where(self._opponent_rusher_mask,action2_1R,action2_1)
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

        slot1_2, pos1_2, action1_2 = self.player1.selection(cstate1_2, self.environment.p1_disposition, warr2_1_type, self.p1_castle.castle_alive, used_p1,self.p1_castle.castle_types)
        slot2_2, pos2_2, action2_2 = p2_training_player.selection(cstate2_2, self.environment.p2_disposition, warr1_1_type, self.p2_castle.castle_alive, used_p2,self.p2_castle.castle_types)
        slot2_2R, pos2_2R, action2_2R = self.playerRusher.selection(cstate2_2, self.environment.p2_disposition, warr1_1_type, self.p2_castle.castle_alive, used_p2,self.p2_castle.castle_types)
        slot2_2= torch.where(self._opponent_rusher_mask,slot2_2R,slot2_2)
        pos2_2 = torch.where(self._opponent_rusher_mask,pos2_2R,pos2_2)
        action2_2 = torch.where(self._opponent_rusher_mask,action2_2R,action2_2)
        
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

        slot1_3, pos1_3, action1_3 = self.player1.selection(cstate1_3, self.environment.p1_disposition, warr2_1_type, self.p1_castle.castle_alive, used_p1,self.p1_castle.castle_types)
        slot2_3, pos2_3, action2_3 = p2_training_player.selection(cstate2_3, self.environment.p2_disposition, warr1_1_type, self.p2_castle.castle_alive, used_p2,self.p2_castle.castle_types)
        slot2_3R, pos2_3R, action2_3R = self.playerRusher.selection(cstate2_3, self.environment.p2_disposition, warr1_1_type, self.p2_castle.castle_alive, used_p2,self.p2_castle.castle_types)
        slot2_3 = torch.where(self._opponent_rusher_mask,slot2_3R,slot2_3)
        pos2_3 = torch.where(self._opponent_rusher_mask,pos2_3R,pos2_3)
        action2_3 = torch.where(self._opponent_rusher_mask,action2_3R,action2_3)
        
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

    def _replay_turn_and_selection(self, player, selection_states, selection_actions, reward_acum, player_name, batch_idx, skip_mask=None) -> None:
        loss_turn = None
        for _ in range(constants.TURN_REPLAYS_PER_BATCH):
            loss_turn = player.replay_turn()

        if self.logger and loss_turn is not None:
            self.logger.log_loss(batch_idx, player.replayed_turn, player_name, "turn", loss_turn)
            if constants.USE_WANDB and loss_turn is not None:
                wandb.log({
                    f"loss/{player_name}_turn": loss_turn,
                    "replayed_turn": player.replayed_turn
                }, step=batch_idx)

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
            action_mask = experience.action_mask[valid]
            next_action_mask = experience.next_action_mask[valid]
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
            action_mask = experience.action_mask
            next_action_mask = experience.next_action_mask

        if states.shape[0] == 0:
            return

        player.remember_turn_batch(
            states, actions, rewards, next_states, dones,
            alive, types, cooldowns, opp_types, next_types, next_alive, next_cooldowns, next_opp_types,
            instance_abilities, next_instance_abilities,
            action_mask, next_action_mask,
        )

    def _build_observations(self):
        speed_p1 = self.environment.speed_por_tipo[self.environment.p1_disposition] / 20.0
        speed_p2 = self.environment.speed_por_tipo[self.environment.p2_disposition] / 20.0

        maxh_p1 = self.environment.max_health_por_tipo[self.environment.p1_disposition]
        maxh_p2 = self.environment.max_health_por_tipo[self.environment.p2_disposition]

        health_norm_p1 = torch.where(maxh_p1 > 0, self.environment.p1_healths / maxh_p1.clamp(min=1e-8), torch.zeros_like(maxh_p1))
        health_norm_p2 = torch.where(maxh_p2 > 0, self.environment.p2_healths / maxh_p2.clamp(min=1e-8), torch.zeros_like(maxh_p2))

        life_p1 = health_norm_p1 * self.environment.p1_alive.float()
        life_p2 = health_norm_p2 * self.environment.p2_alive.float()

        turn_norm = (self.environment.turn_number.float() / constants.MAX_TURNS).clamp(max=1.0)

        obs1 = ObservationV.normalize_batch(
            self.environment.p1_disposition, self.environment.p1_alive, speed_p1, health_norm_p1,
            self.environment.p1_cooldowns, life_p2, self.environment.p2_disposition, turn_norm,
            self.environment.p1_instance_abilities,self.environment.p2_cooldowns,self.environment.p2_instance_abilities,
            self._p2_profile,
        )
        obs2 = ObservationV.normalize_batch(
            self.environment.p2_disposition, self.environment.p2_alive, speed_p2, health_norm_p2,
            self.environment.p2_cooldowns, life_p1, self.environment.p1_disposition, turn_norm,
            self.environment.p2_instance_abilities,self.environment.p1_cooldowns,self.environment.p1_instance_abilities,
            self._p1_profile,
        )
        return obs1, obs2
    
    def _ability_type_masks(self, disposition, instance_abilities):
        """(N,3,4) bool — para cada uno de los 4 botones de habilidad de cada
        slot, si esa habilidad concreta (según el tipo de guerrero y el pool
        equipado) es de tipo ATAQUE o de tipo DEFENSA/CURA. No mira si el
        botón es jugable ahora mismo (eso lo aporta action_mask por fuera)."""
        tipo_expand = disposition.unsqueeze(-1).expand(-1, -1, 4)
        effect_type = self.environment.effect_type_por_tipo_habilidad[tipo_expand, instance_abilities]
        es_ataque = effect_type == EffectType.ATTACK
        es_defensa_cura = (
            (effect_type == EffectType.DEFEND_FULL) | (effect_type == EffectType.DEFEND_HALF) |
            (effect_type == EffectType.SELF_HEAL) | (effect_type == EffectType.TEAM_HEAL)
        )
        return es_ataque, es_defensa_cura

    def _update_profile_accumulators(
        self, p1_types_now, p1_abilities_now, p1_action_mask_now, p1_alive_now,
        p2_types_now, p2_abilities_now, p2_action_mask_now, p2_alive_now,
    ):
        """Acumula, para este turno concreto, cuántos slots tenían ataque
        disponible / defensa-cura disponible (denominador de 'oportunidad'),
        y suma los conteos reales que ya calculó VectorizedEnvironment.turn()
        (self.environment.p1_attacks, etc, ya reseteados a valores de ESTE
        turno). Debe llamarse justo después de self.environment.turn()."""
        p1_es_ataque, p1_es_def = self._ability_type_masks(p1_types_now, p1_abilities_now)
        p1_atk_valid = p1_action_mask_now[:, :, :4] & p1_es_ataque
        p1_def_valid = p1_action_mask_now[:, :, :4] & p1_es_def
        _p1_atk_opp = p1_atk_valid.any(dim=-1).sum(dim=-1).float()
        _p1_def_opp = p1_def_valid.any(dim=-1).sum(dim=-1).float()
        _p1_move_opp = p1_alive_now.sum(dim=-1).float()
        _p1_atk_taken = self.environment.p1_attacks.float()
        _p1_def_taken = self.environment.p1_defenses.float()
        _p1_move_taken = self.environment.p1_movements.float()
        _p1_dmg_dealt = self.environment.p1_damage
        _p1_dmg_recv = self.environment.p2_damage

        p2_es_ataque, p2_es_def = self._ability_type_masks(p2_types_now, p2_abilities_now)
        p2_atk_valid = p2_action_mask_now[:, :, :4] & p2_es_ataque
        p2_def_valid = p2_action_mask_now[:, :, :4] & p2_es_def
        _p2_atk_opp = p2_atk_valid.any(dim=-1).sum(dim=-1).float()
        _p2_def_opp = p2_def_valid.any(dim=-1).sum(dim=-1).float()
        _p2_move_opp = p2_alive_now.sum(dim=-1).float()
        _p2_atk_taken = self.environment.p2_attacks.float()
        _p2_def_taken = self.environment.p2_defenses.float()
        _p2_move_taken = self.environment.p2_movements.float()
        _p2_dmg_dealt = self.environment.p2_damage
        _p2_dmg_recv = self.environment.p1_damage

        self._p1_profile = self._compute_profile(
            self._p1_profile,
            _p1_atk_taken, _p1_atk_opp, _p1_def_taken, _p1_def_opp,
            _p1_move_taken, _p1_move_opp, _p1_dmg_dealt, _p1_dmg_recv,
        )
        self._p2_profile = self._compute_profile(
            self._p2_profile,
            _p2_atk_taken, _p2_atk_opp, _p2_def_taken, _p2_def_opp,
            _p2_move_taken, _p2_move_opp, _p2_dmg_dealt, _p2_dmg_recv,
        )

    def _compute_profile(self, old_profile, atk_taken, atk_opp, def_taken, def_opp, move_taken, move_opp, dmg_dealt, dmg_recv):
        """EMA turno a turno. Si el denominador de oportunidad de este turno es 0
        (nadie tuvo esa opción disponible), no hay observación nueva y se mantiene
        el valor anterior sin mezclar — evita contaminar la EMA con NaN por 0/0."""
        decay = constants.PROFILE_EMA_DECAY

        new_aggression = torch.where(atk_opp > 0, atk_taken / atk_opp.clamp(min=1), old_profile[:, 0])
        aggression = decay * new_aggression + (1 - decay) * old_profile[:, 0]

        new_movement = torch.where(move_opp > 0, move_taken / move_opp.clamp(min=1), old_profile[:, 1])
        movement_freq = decay * new_movement + (1 - decay) * old_profile[:, 1]

        new_defense = torch.where(def_opp > 0, def_taken / def_opp.clamp(min=1), old_profile[:, 2])
        defense_usage = decay * new_defense + (1 - decay) * old_profile[:, 2]

        damage_ratio_raw = dmg_dealt / dmg_recv.clamp(min=1)
        new_damage_ratio = damage_ratio_raw / (damage_ratio_raw + 1.0)
        damage_ratio = decay * new_damage_ratio + (1 - decay) * old_profile[:, 3]

        return torch.stack([aggression, movement_freq, defense_usage, damage_ratio], dim=-1)
    
    def _estimate_aggression_prior(self, disposition, instance_abilities):
        damage = self.environment.damage_por_tipo_habilidad[disposition.unsqueeze(-1),instance_abilities]
        effect_type = self.environment.effect_type_por_tipo_habilidad[disposition.unsqueeze(-1),instance_abilities]
        attack_mask = effect_type == EffectType.ATTACK
        attack_damage = damage * attack_mask
        total_damage = attack_damage.sum(dim=(1,2))
        prior = total_damage / constants.PROFILE_DAMAGE_POTENTIAL_REF
        prior = prior.clamp(0.0, 1.0)
        return prior

    def _turn_mixed_opponent(self, obs2_tensor, from_pool, grouped_opponents, p2_training_player):
        actions = p2_training_player.turn(
            obs2_tensor, self.environment.p2_disposition, self.environment.p2_cooldowns,
            self.environment.p2_alive, self.environment.p1_disposition,
            self.environment.p2_instance_abilities,
        )
        rusher_mask_3 = self._opponent_rusher_mask.unsqueeze(-1)
        actions_rusher = self.playerRusher.turn(obs2_tensor, self.environment.p2_disposition, self.environment.p2_cooldowns,
            self.environment.p2_alive, self.environment.p1_disposition,
            self.environment.p2_instance_abilities,)
        
        actions = torch.where(rusher_mask_3,actions_rusher,actions)

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