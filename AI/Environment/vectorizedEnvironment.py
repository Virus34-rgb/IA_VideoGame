"""
Entorno vectorizado para Castle Game.
"""
import torch
from typing import Tuple, Dict, Any, Optional

from AI.Environment.gameState import GameState
from AI.Environment.resolve_actions import resolveAction
from AI.Environment.statsV import StatsV
from AI.Environment.warriorFactory import get_warriors_classes
from AI.Environment.abilityData import EffectType
import constants


class VectorizedEnvironment:
    def __init__(self, N: int) -> None:
        self.N: int = N
        self.indices: torch.Tensor = torch.arange(N)

        self.warriors_classes: Dict[int, Any] = get_warriors_classes()

        self.max_health_por_tipo: torch.Tensor
        self.speed_por_tipo: torch.Tensor
        self.damage_por_tipo_habilidad: torch.Tensor
        self.turn_cd_por_tipo_habilidad: torch.Tensor
        self.target_mask_por_tipo_habilidad: torch.Tensor
        self.effect_type_por_tipo_habilidad: torch.Tensor
        self._build_static_tables()
        self.resolver = resolveAction(
            self.max_health_por_tipo,
            self.damage_por_tipo_habilidad,
            self.turn_cd_por_tipo_habilidad,
            self.target_mask_por_tipo_habilidad,
            self.effect_type_por_tipo_habilidad
        )

        self.p1_disposition: torch.Tensor
        self.p2_disposition: torch.Tensor
        self.p1_healths: torch.Tensor
        self.p2_healths: torch.Tensor
        self.p1_cooldowns: torch.Tensor
        self.p2_cooldowns: torch.Tensor
        self.p1_alive: torch.Tensor
        self.p2_alive: torch.Tensor
        self.p1_initialWarrior: torch.Tensor
        self.p2_initialWarrior: torch.Tensor
        self.p1_initialPosition: torch.Tensor
        self.p2_initialPosition: torch.Tensor
        self.p1_deaths: torch.Tensor
        self.p2_deaths: torch.Tensor
        self.ended: torch.Tensor
        self.winner: torch.Tensor
        self.turn_number: torch.Tensor
        self.p1_instance_abilities: torch.Tensor   
        self.p2_instance_abilities: torch.Tensor

        self.stats: StatsV = StatsV()
        self.reset()

    def reset(self) -> GameState:
        self.p1_disposition = torch.zeros((self.N, 3), dtype=torch.long)
        self.p2_disposition = torch.zeros((self.N, 3), dtype=torch.long)
        self.p1_healths = torch.zeros((self.N, 3), dtype=torch.float)
        self.p2_healths = torch.zeros((self.N, 3), dtype=torch.float)
        self.p1_cooldowns = torch.zeros((self.N, 3, 4), dtype=torch.long)
        self.p2_cooldowns = torch.zeros((self.N, 3, 4), dtype=torch.long)
        self.p1_alive = torch.zeros((self.N, 3), dtype=torch.bool)
        self.p2_alive = torch.zeros((self.N, 3), dtype=torch.bool)
        self.p1_initialWarrior = torch.zeros(self.N, dtype=torch.long)
        self.p2_initialWarrior = torch.zeros(self.N, dtype=torch.long)
        self.p1_initialPosition = torch.zeros(self.N, dtype=torch.long)
        self.p2_initialPosition = torch.zeros(self.N, dtype=torch.long)
        self.p1_deaths = torch.zeros(self.N, dtype=torch.long)
        self.p2_deaths = torch.zeros(self.N, dtype=torch.long)
        self.ended = torch.zeros(self.N, dtype=torch.bool)
        self.winner = torch.full((self.N,), -1, dtype=torch.long)
        self.turn_number = torch.zeros(self.N, dtype=torch.int)
        self.p1_instance_abilities = torch.zeros((self.N, 3, 4), dtype=torch.long)
        self.p2_instance_abilities = torch.zeros((self.N, 3, 4), dtype=torch.long)
        self.p1_castle_slots = torch.zeros((self.N, 3), dtype=torch.long)
        self.p2_castle_slots = torch.zeros((self.N, 3), dtype=torch.long)

        self.stats.start_batch(self.N)
        return self.get_state()

    def get_state(self) -> GameState:
        return GameState(
            self.p1_disposition, self.p2_disposition,
            self.p1_deaths, self.p2_deaths, self.turn_number,
        )

    def team_selection(
        self, warrior_p1, pos1, warrior_p2, pos2, selected,
        health1, health2, abilities1, abilities2,
    ) -> GameState:
        self._warrior_selected(warrior_p1, pos1, warrior_p2, pos2, selected, health1, health2, abilities1, abilities2)
        return self.get_state()

    def _warrior_selected(
        self, warrior1, pos1, warrior2, pos2, selected,
        health1, health2, abilities1, abilities2,
    ) -> None:
        if selected == 0:
            self.p1_initialWarrior[self.indices] = warrior1
            self.p2_initialWarrior[self.indices] = warrior2
            self.p1_initialPosition[self.indices] = pos1
            self.p2_initialPosition[self.indices] = pos2

        self.p1_disposition[self.indices, pos1] = warrior1
        self.p2_disposition[self.indices, pos2] = warrior2
        self.p1_healths[self.indices, pos1] = health1
        self.p2_healths[self.indices, pos2] = health2
        self.p1_alive[self.indices, pos1] = True
        self.p2_alive[self.indices, pos2] = True
        self.p1_instance_abilities[self.indices, pos1] = abilities1
        self.p2_instance_abilities[self.indices, pos2] = abilities2

        self.stats.accumulate_warrior_use(warrior1, warrior2)

    def turn(self, actionsp1: torch.Tensor, actionsp2: torch.Tensor):
        self.turn_number += 1

        self.p1_cooldowns = torch.where(
            self.p1_alive.unsqueeze(-1), (self.p1_cooldowns - 1).clamp(min=0), self.p1_cooldowns,
        )
        self.p2_cooldowns = torch.where(
            self.p2_alive.unsqueeze(-1), (self.p2_cooldowns - 1).clamp(min=0), self.p2_cooldowns,
        )

        ya_terminadas_antes = self.ended.clone()

        order, actor_alive_inicio = self._get_turn_order()
        p1_alive_inicio = actor_alive_inicio[:, :3]
        p2_alive_inicio = actor_alive_inicio[:, 3:]

        damage_p1 = torch.zeros(self.N)
        damage_p2 = torch.zeros(self.N)
        damage_avoided_p1 = torch.zeros(self.N)
        damage_avoided_p2 = torch.zeros(self.N)
        blocks_p1 = torch.zeros(self.N)
        blocks_p2 = torch.zeros(self.N)
        heal_p1 = torch.zeros(self.N)
        heal_p2 = torch.zeros(self.N)
        wasted_heal_p1 = torch.zeros(self.N)
        wasted_heal_p2 = torch.zeros(self.N)
        wasted_defense_p1 = torch.zeros(self.N)
        wasted_defense_p2 = torch.zeros(self.N)
        strategic_movement_p1 = torch.zeros(self.N)
        strategic_movement_p2 = torch.zeros(self.N)
        overkill_damage_p1 = torch.zeros(self.N)
        overkill_damage_p2 = torch.zeros(self.N)
        kill_confirmed_p1 = torch.zeros(self.N)
        kill_confirmed_p2 = torch.zeros(self.N)
        
        p1_health_before = self._normalized_team_health(self.p1_healths, self.p1_disposition)
        p2_health_before = self._normalized_team_health(self.p2_healths, self.p2_disposition)

        for position in range(6):
            actor_idx = order[:, position]
            player = actor_idx // 3
            pos = actor_idx % 3
            es_p1 = (player == 0)

            player_mask = es_p1.unsqueeze(1)
            player_mask_3 = player_mask.unsqueeze(-1)

            own_disp = torch.where(player_mask, self.p1_disposition, self.p2_disposition)
            enemy_disp = torch.where(player_mask, self.p2_disposition, self.p1_disposition)
            own_health = torch.where(player_mask, self.p1_healths, self.p2_healths)
            enemy_health = torch.where(player_mask, self.p2_healths, self.p1_healths)
            own_actions = torch.where(player_mask, actionsp1, actionsp2)
            actor_action = own_actions.gather(1, pos.unsqueeze(1)).squeeze(1)
            enemy_actions = torch.where(player_mask, actionsp2, actionsp1)
            own_alive = torch.where(player_mask, self.p1_alive, self.p2_alive)
            enemy_alive = torch.where(player_mask, self.p2_alive, self.p1_alive)
            own_cooldowns = torch.where(player_mask_3, self.p1_cooldowns, self.p2_cooldowns)
            own_instance_abilities = torch.where(player_mask_3, self.p1_instance_abilities, self.p2_instance_abilities)
            enemy_instance_abilities = torch.where(player_mask_3, self.p2_instance_abilities, self.p1_instance_abilities)
            actor_type = own_disp.gather(1, pos.unsqueeze(1)).squeeze(1)

            # Obtener castle_slots del actor y enemigo
            own_castle_slots = torch.where(player_mask, self.p1_castle_slots, self.p2_castle_slots)
            enemy_castle_slots = torch.where(player_mask, self.p2_castle_slots, self.p1_castle_slots)

            (
                dmg, avoided, blocked, moved, healed,
                new_own_disp, new_enemy_disp, new_own_health, new_enemy_health,
                new_own_cd, new_own_alive, new_enemy_alive,
                new_own_abilities, ability_pool_idx,
                new_own_castle,wasted_heal,defense_wasted,strategic_movement,
                overkill_damage,kill_confirmed
            ) = self.resolver.resolve_action(
                pos, actor_type, own_disp, enemy_disp, own_health, enemy_health,
                own_cooldowns, own_alive, enemy_alive, actor_action, enemy_actions,
                own_instance_abilities, enemy_instance_abilities,
                own_castle_slots, enemy_castle_slots,
            )

            self.p1_disposition = torch.where(player_mask, new_own_disp, new_enemy_disp)
            self.p2_disposition = torch.where(player_mask, new_enemy_disp, new_own_disp)
            self.p1_healths = torch.where(player_mask, new_own_health, new_enemy_health)
            self.p2_healths = torch.where(player_mask, new_enemy_health, new_own_health)
            self.p1_alive = torch.where(player_mask, new_own_alive, new_enemy_alive)
            self.p2_alive = torch.where(player_mask, new_enemy_alive, new_own_alive)
            self.p1_cooldowns = torch.where(player_mask_3, new_own_cd, self.p1_cooldowns)
            self.p2_cooldowns = torch.where(~player_mask_3, new_own_cd, self.p2_cooldowns)
            self.p1_instance_abilities = torch.where(player_mask_3, new_own_abilities, self.p1_instance_abilities)
            self.p2_instance_abilities = torch.where(~player_mask_3, new_own_abilities, self.p2_instance_abilities)
            self.p1_castle_slots = torch.where(player_mask, new_own_castle, self.p1_castle_slots)
            self.p2_castle_slots = torch.where(~player_mask, new_own_castle, self.p2_castle_slots)

            damage_p1 += torch.where(es_p1, dmg, torch.zeros_like(dmg))
            damage_p2 += torch.where(es_p1, torch.zeros_like(dmg), dmg)
            damage_avoided_p1 += torch.where(~es_p1, avoided, torch.zeros_like(avoided))
            damage_avoided_p2 += torch.where(~es_p1, torch.zeros_like(avoided), avoided)
            blocks_p1 += torch.where(~es_p1, blocked, torch.zeros_like(blocked))
            blocks_p2 += torch.where(~es_p1, torch.zeros_like(blocked), blocked)
            heal_p1 += torch.where(es_p1, healed, torch.zeros_like(healed))
            heal_p2 += torch.where(~es_p1, healed, torch.zeros_like(healed))
            heal_p1 += torch.where(es_p1, healed, torch.zeros_like(healed))
            heal_p2 += torch.where(~es_p1, healed, torch.zeros_like(healed))
            wasted_heal_p1 += torch.where(es_p1, wasted_heal, torch.zeros_like(healed))
            wasted_heal_p2 += torch.where(~es_p1, wasted_heal, torch.zeros_like(healed))
            wasted_defense_p1 += torch.where(es_p1, defense_wasted, torch.zeros_like(healed))
            wasted_defense_p2 += torch.where(~es_p1, defense_wasted, torch.zeros_like(healed))
            strategic_movement_p1 += torch.where(es_p1, strategic_movement, torch.zeros_like(strategic_movement))
            strategic_movement_p2 += torch.where(~es_p1, strategic_movement, torch.zeros_like(strategic_movement))
            overkill_damage_p1 += torch.where(es_p1, overkill_damage, torch.zeros_like(overkill_damage))
            overkill_damage_p2 += torch.where(~es_p1, overkill_damage, torch.zeros_like(overkill_damage))
            kill_confirmed_p1 += torch.where(es_p1, kill_confirmed, torch.zeros_like(kill_confirmed))
            kill_confirmed_p2 += torch.where(~es_p1, kill_confirmed, torch.zeros_like(kill_confirmed))

            self.stats.accumulate_movements(moved, es_p1, ~ya_terminadas_antes)
            self.stats.accumulate_attacks(actor_type, ability_pool_idx, es_p1, ~ya_terminadas_antes)

        p1_health_after = self._normalized_team_health(self.p1_healths, self.p1_disposition)
        p2_health_after = self._normalized_team_health(self.p2_healths, self.p2_disposition)
        health_diff_before = p1_health_before / 3 - p2_health_before / 3
        health_diff_after = p1_health_after / 3 - p2_health_after / 3

        p1_new_deaths = (p1_alive_inicio & ~self.p1_alive).sum(dim=1).to(self.p1_deaths.dtype)
        p2_new_deaths = (p2_alive_inicio & ~self.p2_alive).sum(dim=1).to(self.p2_deaths.dtype)

        self.stats.accumulate_turn(
            damage_p1, damage_p2, blocks_p1, blocks_p2,
            damage_avoided_p1, damage_avoided_p2, heal_p1, heal_p2,
            wasted_heal_p1,wasted_heal_p2,wasted_defense_p1,wasted_defense_p2,
            strategic_movement_p1, strategic_movement_p2,
            overkill_damage_p1, overkill_damage_p2,
            kill_confirmed_p1, kill_confirmed_p2,
            ya_terminadas_antes,
        )
        

        rewardP1, rewardP2 = self._calculate_rewards(
            damage_p1, damage_p2, damage_avoided_p1, damage_avoided_p2,
            heal_p1, heal_p2, health_diff_before, health_diff_after,
            p1_new_deaths, p2_new_deaths,wasted_heal_p1,wasted_heal_p2,
            wasted_defense_p1,wasted_defense_p2,strategic_movement_p1,strategic_movement_p2,
            overkill_damage_p1,overkill_damage_p2,kill_confirmed_p1,kill_confirmed_p2,
        )

        rewardP1 = torch.where(ya_terminadas_antes, torch.zeros_like(rewardP1), rewardP1)
        rewardP2 = torch.where(ya_terminadas_antes, torch.zeros_like(rewardP2), rewardP2)

        return self.get_state(), rewardP1, rewardP2, self.ended


    def _get_turn_order(self):
        actor_types = torch.cat([self.p1_disposition, self.p2_disposition], dim=1)
        actor_alive = torch.cat([self.p1_alive, self.p2_alive], dim=1)
        speeds = self.speed_por_tipo[actor_types]
        speeds = torch.where(actor_alive, speeds, torch.full_like(speeds, float("-inf")))
        order = torch.argsort(speeds, dim=1, descending=True, stable=True)
        return order, actor_alive

    def _check_end_conditions(self) -> None:
        ya_terminadas = self.ended.clone()
        max_deaths = constants.MAX_DEATHS_PER_TEAM
        max_turns = constants.MAX_TURNS

        ambos_muertos = (self.p1_deaths >= max_deaths) & (self.p2_deaths >= max_deaths)
        solo_p1_muerto = (self.p1_deaths >= max_deaths) & ~ambos_muertos
        solo_p2_muerto = (self.p2_deaths >= max_deaths) & ~ambos_muertos & ~solo_p1_muerto
        por_turnos = (self.turn_number > max_turns) & ~(ambos_muertos | solo_p1_muerto | solo_p2_muerto)

        p1_health_norm = self._normalized_team_health(self.p1_healths, self.p1_disposition)
        p2_health_norm = self._normalized_team_health(self.p2_healths, self.p2_disposition)

        p1_menos_bajas = self.p1_deaths < self.p2_deaths
        p2_menos_bajas = self.p2_deaths < self.p1_deaths
        bajas_iguales = ~p1_menos_bajas & ~p2_menos_bajas

        HEALTH_EPS = 1e-4
        p1_mas_vida = bajas_iguales & (p1_health_norm > p2_health_norm + HEALTH_EPS)
        p2_mas_vida = bajas_iguales & (p2_health_norm > p1_health_norm + HEALTH_EPS)

        por_turnos_gana_p1 = por_turnos & (p1_menos_bajas | p1_mas_vida)
        por_turnos_gana_p2 = por_turnos & (p2_menos_bajas | p2_mas_vida)
        por_turnos_empate = por_turnos & ~(por_turnos_gana_p1 | por_turnos_gana_p2)

        new_winner = self.winner.clone()
        new_winner = torch.where(ambos_muertos, torch.full_like(new_winner, 2), new_winner)
        new_winner = torch.where(solo_p1_muerto, torch.full_like(new_winner, 1), new_winner)
        new_winner = torch.where(solo_p2_muerto, torch.full_like(new_winner, 0), new_winner)
        new_winner = torch.where(por_turnos_gana_p1, torch.full_like(new_winner, 0), new_winner)
        new_winner = torch.where(por_turnos_gana_p2, torch.full_like(new_winner, 1), new_winner)
        new_winner = torch.where(por_turnos_empate, torch.full_like(new_winner, 2), new_winner)

        termina_ahora = (ambos_muertos | solo_p1_muerto | solo_p2_muerto | por_turnos) & ~ya_terminadas

        self.winner = torch.where(termina_ahora, new_winner, self.winner)
        self.ended = self.ended | termina_ahora

        self.stats.close_finished_games(
            termina_ahora, self.winner, self.p1_deaths, self.p2_deaths, self.turn_number,
            por_muerte_mask=(ambos_muertos | solo_p1_muerto | solo_p2_muerto),
            por_turnos_mask=por_turnos,
        )

    def _normalized_team_health(self, healths, disposition):
        # healths: (N, 3), disposition: (N, 3)
        # max_health_por_tipo[disposition] es (N, 3)
        max_health = self.max_health_por_tipo[disposition]
        # Para slots vacíos (disposition == 0), max_health será 0 (porque la tabla tiene 0 en la fila 0).
        # Dividir por 0 da inf. Hay que enmascarar.
        alive = disposition > 0
        norm_health = torch.zeros_like(healths)
        norm_health[alive] = healths[alive] / max_health[alive]
        return norm_health.sum(dim=1)

    def _turn_penalty(self) -> torch.Tensor:
        turn = self.turn_number.float()
        exceso = (turn - constants.TURN_PENALTY_RAMP_START).clamp(min=0.0)
        progresion = (exceso / constants.TURN_PENALTY_RAMP_TURNS).clamp(max=1.0)
        return constants.TURN_PENALTY_BASE + progresion * (constants.TURN_PENALTY_MAX - constants.TURN_PENALTY_BASE)

    def _reward(self, **components: torch.Tensor) -> torch.Tensor:
        weighted = sum(constants.REWARD_WEIGHTS[name] * value for name, value in components.items())
        return (weighted - self._turn_penalty()) / constants.REWARD_SCALE

    def _calculate_rewards(
        self, damage_p1, damage_p2, damage_avoided_p1, damage_avoided_p2,
        healed_p1, healed_p2, health_diff_before, health_diff_after, newDeaths_p1, newDeaths_p2,
        wasted_heal_p1,wasted_heal_p2,wasted_defense_p1,wasted_defense_p2,strategic_movement_p1,strategic_movement_p2,
        overkill_damage_p1, overkill_damage_p2,kill_confirmed_p1, kill_confirmed_p2
    ):
        self.p1_deaths += newDeaths_p1
        self.p2_deaths += newDeaths_p2
        self._check_end_conditions()

        gano_p1 = self.winner == 0
        gano_p2 = self.winner == 1
        empate = self.winner == 2
        win_p1 = torch.where(
            gano_p1, torch.full_like(damage_p1, constants.WIN_REWARD),
            torch.where(gano_p2, torch.full_like(damage_p1, -constants.WIN_REWARD),
                torch.where(empate, torch.full_like(damage_p1, -constants.DRAW_PENALTY), torch.zeros_like(damage_p1))),
        )
        win_p2 = torch.where(
            gano_p2, torch.full_like(damage_p1, constants.WIN_REWARD),
            torch.where(gano_p1, torch.full_like(damage_p1, -constants.WIN_REWARD),
                torch.where(empate, torch.full_like(damage_p1, -constants.DRAW_PENALTY), torch.zeros_like(damage_p1))),
        )

        shaping_term_p1 = constants.DISCOUNT_FACTOR * health_diff_after - health_diff_before
        shaping_term_p2 = -shaping_term_p1

        rewardP1 = self._reward(
            damage=damage_p1 - damage_p2, deaths=newDeaths_p2 - newDeaths_p1, win=win_p1,
            blocks=damage_avoided_p1, heal=healed_p1, shaping_weight=shaping_term_p1,
            wasted_heal = wasted_heal_p1,wasted_defense = wasted_defense_p1,
            strategic_movement = strategic_movement_p1,kill_confirmed = kill_confirmed_p1, overkill_damage = overkill_damage_p1
        )
        rewardP2 = self._reward(
            damage=damage_p2 - damage_p1, deaths=newDeaths_p1 - newDeaths_p2, win=win_p2,
            blocks=damage_avoided_p2, heal=healed_p2, shaping_weight=shaping_term_p2,
            wasted_heal = wasted_heal_p2,wasted_defense = wasted_defense_p2,
            strategic_movement = strategic_movement_p2,kill_confirmed = kill_confirmed_p2, overkill_damage = overkill_damage_p2
        )
        return rewardP1, rewardP2

    def _build_static_tables(self) -> None:
        num_types = max(self.warriors_classes.keys()) + 1
        num_abilities = constants.MAX_POOL_SIZE
        num_slots = constants.NUM_SLOTS

        max_health_por_tipo = torch.zeros(num_types, dtype=torch.float)
        speed_por_tipo = torch.zeros(num_types, dtype=torch.float)
        damage_por_tipo_habilidad = torch.zeros(num_types, num_abilities, dtype=torch.float)
        turn_cd_por_tipo_habilidad = torch.zeros(num_types, num_abilities, dtype=torch.long)
        target_mask_por_tipo_habilidad = torch.zeros(num_types, num_abilities, num_slots, dtype=torch.bool)
        effect_type_por_tipo_habilidad = torch.zeros(num_types, num_abilities, dtype=torch.long)

        for warrior_id, warrior_data in self.warriors_classes.items():
            max_health_por_tipo[warrior_id] = warrior_data.max_health
            speed_por_tipo[warrior_id] = warrior_data.speed
            for ability_idx, ability in enumerate(warrior_data.ability_pool):
                damage_por_tipo_habilidad[warrior_id, ability_idx] = ability.damage
                turn_cd_por_tipo_habilidad[warrior_id, ability_idx] = ability.turn_cd
                effect_type_por_tipo_habilidad[warrior_id, ability_idx] = int(ability.effect_type)
                for target_pos in ability.target_positions:
                    target_mask_por_tipo_habilidad[warrior_id, ability_idx, target_pos] = True

        self.max_health_por_tipo = max_health_por_tipo
        self.speed_por_tipo = speed_por_tipo
        self.damage_por_tipo_habilidad = damage_por_tipo_habilidad
        self.turn_cd_por_tipo_habilidad = turn_cd_por_tipo_habilidad
        self.target_mask_por_tipo_habilidad = target_mask_por_tipo_habilidad
        self.effect_type_por_tipo_habilidad = effect_type_por_tipo_habilidad