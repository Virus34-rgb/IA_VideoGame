"""
Entorno vectorizado para Castle Game.
"""
import torch
from typing import Tuple, Dict, Any, Optional

from AI.Environment.gameState import GameState
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

            (
                dmg, avoided, blocked, moved, healed,
                new_own_disp, new_enemy_disp, new_own_health, new_enemy_health,
                new_own_cd, new_own_alive, new_enemy_alive,
                new_own_abilities,   # NUEVO
            ) = self._resolve_action(
                pos, actor_type, own_disp, enemy_disp, own_health, enemy_health,
                own_cooldowns, own_alive, enemy_alive, actor_action, enemy_actions,
                own_instance_abilities, enemy_instance_abilities,
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

            damage_p1 += torch.where(es_p1, dmg, torch.zeros_like(dmg))
            damage_p2 += torch.where(es_p1, torch.zeros_like(dmg), dmg)
            damage_avoided_p1 += torch.where(~es_p1, avoided, torch.zeros_like(avoided))
            damage_avoided_p2 += torch.where(~es_p1, torch.zeros_like(avoided), avoided)
            blocks_p1 += torch.where(~es_p1, blocked, torch.zeros_like(blocked))
            blocks_p2 += torch.where(~es_p1, torch.zeros_like(blocked), blocked)
            heal_p1 += torch.where(es_p1, healed, torch.zeros_like(healed))
            heal_p2 += torch.where(~es_p1, healed, torch.zeros_like(healed))
            
            own_slot_abilities = own_instance_abilities.gather(1, pos.view(-1, 1, 1).expand(-1, 1, 4)).squeeze(1)  # (N,4)
            ability_pool_idx = own_slot_abilities.gather(1, actor_action.clamp(0, 3).unsqueeze(1)).squeeze(1)     # (N,)

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
            damage_avoided_p1, damage_avoided_p2, heal_p1, heal_p2, ya_terminadas_antes,
        )

        rewardP1, rewardP2 = self._calculate_rewards(
            damage_p1, damage_p2, damage_avoided_p1, damage_avoided_p2,
            heal_p1, heal_p2, health_diff_before, health_diff_after, p1_new_deaths, p2_new_deaths,
        )

        rewardP1 = torch.where(ya_terminadas_antes, torch.zeros_like(rewardP1), rewardP1)
        rewardP2 = torch.where(ya_terminadas_antes, torch.zeros_like(rewardP2), rewardP2)

        return self.get_state(), rewardP1, rewardP2, self.ended

    def _resolve_action(
        self, pos, actors, own_disposition, enemy_disposition, own_health, enemy_health,
        own_cooldowns, own_alive, enemy_alive, actions_actor, enemy_actions,
        own_instance_abilities, enemy_instance_abilities,   # NUEVO
    ):
        actor_alive_now = own_alive.gather(1, pos.unsqueeze(1)).squeeze(1)

        own_slot_abilities = own_instance_abilities.gather(1, pos.view(-1, 1, 1).expand(-1, 1, 4)).squeeze(1)  # (N,4)
        ability_pool_idx = own_slot_abilities.gather(1, actions_actor.clamp(0, 3).unsqueeze(1)).squeeze(1)     # (N,)
        effect_type = self.effect_type_por_tipo_habilidad[actors, ability_pool_idx]                            # (N,)

        es_habilidad = (actions_actor >= 0) & (actions_actor <= 3)   # NUEVO: excluye movimiento(5,6) y muerto(-1)

        mask_movPos = (actions_actor == 5) & (pos != 2) & actor_alive_now
        mask_movNeg = (actions_actor == 6) & (pos != 0) & actor_alive_now
        mask_self_heal = es_habilidad & (effect_type == EffectType.SELF_HEAL) & actor_alive_now
        mask_team_heal = es_habilidad & (effect_type == EffectType.TEAM_HEAL) & actor_alive_now
        mask_defend = es_habilidad & (effect_type == EffectType.DEFEND) & actor_alive_now
        mask_ataque = es_habilidad & (effect_type == EffectType.ATTACK) & actor_alive_now

        moved, new_disp_mov, new_health_mov, new_cd_mov, new_abilities_mov = self._resolve_action_movement(
            actors, own_disposition, own_health, own_cooldowns, own_instance_abilities, actions_actor, pos
        )

        own_new_disp = own_disposition.clone()
        own_new_disp = torch.where(mask_movPos.unsqueeze(1), new_disp_mov, own_new_disp)
        own_new_disp = torch.where(mask_movNeg.unsqueeze(1), new_disp_mov, own_new_disp)

        damage_raw, blocked_raw, enemy_health_after_attack, enemy_alive_after_attack = self._resolve_action_attack(
            actors, ability_pool_idx, enemy_disposition, enemy_health, enemy_alive, enemy_actions, enemy_instance_abilities,
        )
        damage = torch.where(mask_ataque, damage_raw, torch.zeros_like(damage_raw))
        blocked = torch.where(mask_ataque, blocked_raw, torch.zeros_like(blocked_raw))

        healed_self, own_health_self = self._resolve_action_self_heal(actors, ability_pool_idx, pos, own_health)
        healed_team, own_health_team = self._resolve_action_team_heal(actors, ability_pool_idx, own_disposition, own_health, own_alive)

        own_new_health = own_health.clone()
        own_new_health = torch.where(mask_movPos.unsqueeze(1), new_health_mov, own_new_health)
        own_new_health = torch.where(mask_movNeg.unsqueeze(1), new_health_mov, own_new_health)
        own_new_health = torch.where(mask_self_heal.unsqueeze(1), own_health_self, own_new_health)
        own_new_health = torch.where(mask_team_heal.unsqueeze(1), own_health_team, own_new_health)
        enemy_new_health = torch.where(mask_ataque.unsqueeze(1), enemy_health_after_attack, enemy_health)

        mask_usa_habilidad = (mask_ataque | mask_self_heal | mask_team_heal | mask_defend)
        own_cd_new = self._update_own_cooldowns(actors, actions_actor, ability_pool_idx, pos, own_cooldowns, mask_usa_habilidad)

        mask_movPos_4 = mask_movPos.view(-1, 1, 1)
        mask_movNeg_4 = mask_movNeg.view(-1, 1, 1)
        own_cd_new = torch.where(mask_movNeg_4, new_cd_mov, own_cd_new)
        own_cd_new = torch.where(mask_movPos_4, new_cd_mov, own_cd_new)

        own_abilities_new = torch.where(mask_movNeg_4, new_abilities_mov, own_instance_abilities)
        own_abilities_new = torch.where(mask_movPos_4, new_abilities_mov, own_abilities_new)

        enemy_alive_final = torch.where(mask_ataque.unsqueeze(1), enemy_alive_after_attack, enemy_alive)

        heal = torch.zeros_like(own_health[:, 0])
        heal = torch.where(mask_self_heal, healed_self, heal)
        heal = torch.where(mask_team_heal, healed_team, heal)

        damage_avoided = torch.where(mask_ataque, blocked_raw, torch.zeros_like(blocked_raw))

        return (
            damage, damage_avoided, blocked, moved, heal,
            own_new_disp, enemy_disposition, own_new_health, enemy_new_health,
            own_cd_new, own_alive, enemy_alive_final,
            own_abilities_new,
        )

    def _resolve_action_movement(
        self, actors, own_disposition, own_health, own_cooldowns, own_instance_abilities, actions_actor, pos,
    ):
        mask_movPos = (actions_actor == 5) & (pos != 2)
        mask_movNeg = (actions_actor == 6) & (pos != 0)
        moved = (mask_movPos | mask_movNeg).float()

        pos_destino_pos = (pos + 1).clamp(max=2)
        pos_destino_neg = (pos - 1).clamp(min=0)

        own_new_disp = own_disposition.clone()
        origen = own_disposition.gather(1, pos.unsqueeze(1)).squeeze(1)
        destino = own_disposition.gather(1, pos_destino_pos.unsqueeze(1)).squeeze(1)
        own_new_disp.scatter_(1, pos.unsqueeze(1), torch.where(mask_movPos, destino, origen).unsqueeze(1))
        own_new_disp.scatter_(1, pos_destino_pos.unsqueeze(1), torch.where(mask_movPos, origen, destino).unsqueeze(1))

        origen2 = own_new_disp.gather(1, pos.unsqueeze(1)).squeeze(1)
        destino2 = own_new_disp.gather(1, pos_destino_neg.unsqueeze(1)).squeeze(1)
        own_new_disp_final = own_new_disp.clone()
        own_new_disp_final.scatter_(1, pos.unsqueeze(1), torch.where(mask_movNeg, destino2, origen2).unsqueeze(1))
        own_new_disp_final.scatter_(1, pos_destino_neg.unsqueeze(1), torch.where(mask_movNeg, origen2, destino2).unsqueeze(1))

        own_new_health = own_health.clone()
        origen_h = own_health.gather(1, pos.unsqueeze(1)).squeeze(1)
        destino_h = own_health.gather(1, pos_destino_pos.unsqueeze(1)).squeeze(1)
        own_new_health.scatter_(1, pos.unsqueeze(1), torch.where(mask_movPos, destino_h, origen_h).unsqueeze(1))
        own_new_health.scatter_(1, pos_destino_pos.unsqueeze(1), torch.where(mask_movPos, origen_h, destino_h).unsqueeze(1))

        origen_h2 = own_new_health.gather(1, pos.unsqueeze(1)).squeeze(1)
        destino_h2 = own_new_health.gather(1, pos_destino_neg.unsqueeze(1)).squeeze(1)
        own_new_health_final = own_new_health.clone()
        own_new_health_final.scatter_(1, pos.unsqueeze(1), torch.where(mask_movNeg, destino_h2, origen_h2).unsqueeze(1))
        own_new_health_final.scatter_(1, pos_destino_neg.unsqueeze(1), torch.where(mask_movNeg, origen_h2, destino_h2).unsqueeze(1))

        pos_e = pos.view(-1, 1, 1).expand(-1, 1, 4)
        pos_destino_pos_e = pos_destino_pos.view(-1, 1, 1).expand(-1, 1, 4)
        pos_destino_neg_e = pos_destino_neg.view(-1, 1, 1).expand(-1, 1, 4)
        mask_movPos_4 = mask_movPos.view(-1, 1, 1).expand(-1, 1, 4)
        mask_movNeg_4 = mask_movNeg.view(-1, 1, 1).expand(-1, 1, 4)

        own_new_cd = own_cooldowns.clone()
        origen_cd = own_cooldowns.gather(1, pos_e)
        destino_cd = own_cooldowns.gather(1, pos_destino_pos_e)
        own_new_cd.scatter_(1, pos_e, torch.where(mask_movPos_4, destino_cd, origen_cd))
        own_new_cd.scatter_(1, pos_destino_pos_e, torch.where(mask_movPos_4, origen_cd, destino_cd))

        origen_cd2 = own_new_cd.gather(1, pos_e)
        destino_cd2 = own_new_cd.gather(1, pos_destino_neg_e)
        own_new_cd_final = own_new_cd.clone()
        own_new_cd_final.scatter_(1, pos_e, torch.where(mask_movNeg_4, destino_cd2, origen_cd2))
        own_new_cd_final.scatter_(1, pos_destino_neg_e, torch.where(mask_movNeg_4, origen_cd2, destino_cd2))

        own_new_abilities = own_instance_abilities.clone()
        origen_ab = own_instance_abilities.gather(1, pos_e)
        destino_ab = own_instance_abilities.gather(1, pos_destino_pos_e)
        own_new_abilities.scatter_(1, pos_e, torch.where(mask_movPos_4, destino_ab, origen_ab))
        own_new_abilities.scatter_(1, pos_destino_pos_e, torch.where(mask_movPos_4, origen_ab, destino_ab))

        origen_ab2 = own_new_abilities.gather(1, pos_e)
        destino_ab2 = own_new_abilities.gather(1, pos_destino_neg_e)
        own_new_abilities_final = own_new_abilities.clone()
        own_new_abilities_final.scatter_(1, pos_e, torch.where(mask_movNeg_4, destino_ab2, origen_ab2))
        own_new_abilities_final.scatter_(1, pos_destino_neg_e, torch.where(mask_movNeg_4, origen_ab2, destino_ab2))

        return moved, own_new_disp_final, own_new_health_final, own_new_cd_final, own_new_abilities_final

    def _resolve_action_attack(
        self, actors, ability_pool_idx, enemy_disposition, enemy_health, enemy_alive, enemy_actions,
        enemy_instance_abilities,
    ):
        would_be_damage = self.damage_por_tipo_habilidad[actors, ability_pool_idx]
        target_mask = self.target_mask_por_tipo_habilidad[actors, ability_pool_idx]

        enemy_ability_pool_idx = enemy_instance_abilities.gather(
            2, enemy_actions.clamp(0, 3).unsqueeze(-1)
        ).squeeze(-1)  # (N,3)
        enemy_effect_type = self.effect_type_por_tipo_habilidad[enemy_disposition, enemy_ability_pool_idx]  # (N,3)
        enemy_es_habilidad = (enemy_actions >= 0) & (enemy_actions <= 3)  # (N,3)

        enemy_new_health = enemy_health.clone()
        damage_total = torch.zeros_like(would_be_damage)
        avoided_total = torch.zeros_like(would_be_damage)
        blocks_total = torch.zeros_like(would_be_damage)

        for slot in range(3):
            es_target = target_mask[:, slot] & enemy_alive[:, slot]
            enemy_id_slot = enemy_disposition[:, slot]
            enemy_defending = (enemy_effect_type[:, slot] == EffectType.DEFEND) & enemy_es_habilidad[:, slot]  # NUEVO

            full_block = ((enemy_id_slot == 1) | (enemy_id_slot == 3)) & enemy_defending
            half_block = (enemy_id_slot == 5) & enemy_defending

            hit_damage = torch.where(
                full_block, torch.zeros_like(would_be_damage),
                torch.where(half_block, would_be_damage / 2, would_be_damage),
            )
            avoided = torch.where(
                full_block, would_be_damage,
                torch.where(half_block, would_be_damage / 2, torch.zeros_like(would_be_damage)),
            )
            blocked_flag = (full_block | half_block).float()

            hit_damage = torch.where(es_target, hit_damage, torch.zeros_like(hit_damage))
            avoided = torch.where(es_target, avoided, torch.zeros_like(avoided))
            blocked_flag = torch.where(es_target, blocked_flag, torch.zeros_like(blocked_flag))

            health_slot_actual = enemy_new_health[:, slot]
            enemy_new_health[:, slot] = torch.where(es_target, health_slot_actual - hit_damage, health_slot_actual)

            damage_total += hit_damage
            avoided_total += avoided
            blocks_total += blocked_flag

        enemy_new_alive = enemy_alive & (enemy_new_health > 0)
        return damage_total, blocks_total, enemy_new_health, enemy_new_alive

    def _resolve_action_self_heal(self, actors, ability_pool_idx, pos, own_health):
        heal_amount = self.damage_por_tipo_habilidad[actors, ability_pool_idx]
        max_health_actor = self.max_health_por_tipo[actors]
        current = own_health.gather(1, pos.unsqueeze(1)).squeeze(1)
        new_value = torch.min(max_health_actor, current + heal_amount)
        healed = new_value - current
        own_new_health = own_health.scatter(1, pos.unsqueeze(1), new_value.unsqueeze(1))
        return healed, own_new_health

    def _resolve_action_team_heal(self, actors, ability_pool_idx, own_disposition, own_health, own_alive):
        heal_amount = self.damage_por_tipo_habilidad[actors, ability_pool_idx].unsqueeze(1)
        max_health_slot = self.max_health_por_tipo[own_disposition]
        new_value = torch.min(max_health_slot, own_health + heal_amount)
        healed_per_slot = torch.where(own_alive, new_value - own_health, torch.zeros_like(own_health))
        own_new_health = own_health + healed_per_slot
        total_healed = healed_per_slot.sum(dim=1)
        return total_healed, own_new_health

    def _update_own_cooldowns(self, actors, accion_actor, ability_pool_idx, pos, own_cooldowns, mask_usa_habilidad):
        """
        Fija el cooldown del botón usado a turn_cd de la habilidad real (vía ability_pool_idx).
        El decremento por turno ya se aplicó de forma centralizada en turn().
        """
        turns_cd = self.turn_cd_por_tipo_habilidad[actors, ability_pool_idx] 

        slot_expand = pos.view(-1, 1, 1).expand(-1, 1, 4)
        actor_cd = own_cooldowns.gather(1, slot_expand).squeeze(1)
        button_onehot = torch.nn.functional.one_hot(accion_actor.clamp(0, 3), num_classes=4).bool()

        turns_cd_expand = turns_cd.unsqueeze(1).expand(-1, 4)
        marked = torch.where(button_onehot, turns_cd_expand, actor_cd)
        new_actor_cd = torch.where(mask_usa_habilidad.unsqueeze(1), marked, actor_cd)

        return own_cooldowns.scatter(1, slot_expand, new_actor_cd.unsqueeze(1))

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
        return (healths / self.max_health_por_tipo[disposition]).sum(dim=1)

    def _turn_penalty(self) -> torch.Tensor:
        turn = self.turn_number.float()
        exceso = (turn - constants.TURN_PENALTY_RAMP_START).clamp(min=0.0)
        progresion = (exceso / constants.TURN_PENALTY_RAMP_TURNS).clamp(max=1.0)
        return constants.TURN_PENALTY_BASE + progresion * (constants.TURN_PENALTY_MAX - constants.TURN_PENALTY_BASE)

    def _reward(self, **components: torch.Tensor) -> torch.Tensor:
        weighted = sum(constants.REWARD_WEIGHTS[name] * value for name, value in components.items())
        return weighted - self._turn_penalty()

    def _calculate_rewards(
        self, damage_p1, damage_p2, damage_avoided_p1, damage_avoided_p2,
        healed_p1, healed_p2, health_diff_before, health_diff_after, newDeaths_p1, newDeaths_p2,
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
        )
        rewardP2 = self._reward(
            damage=damage_p2 - damage_p1, deaths=newDeaths_p1 - newDeaths_p2, win=win_p2,
            blocks=damage_avoided_p2, heal=healed_p2, shaping_weight=shaping_term_p2,
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
        effect_type_por_tipo_habilidad = torch.zeros(num_types, num_abilities, dtype=torch.long)  # NUEVO

        for warrior_id, warrior_data in self.warriors_classes.items():
            max_health_por_tipo[warrior_id] = warrior_data.max_health
            speed_por_tipo[warrior_id] = warrior_data.speed
            for ability_idx, ability in enumerate(warrior_data.ability_pool):
                damage_por_tipo_habilidad[warrior_id, ability_idx] = ability.damage
                turn_cd_por_tipo_habilidad[warrior_id, ability_idx] = ability.turn_cd
                effect_type_por_tipo_habilidad[warrior_id, ability_idx] = int(ability.effect_type)  # NUEVO
                for target_pos in ability.target_positions:
                    target_mask_por_tipo_habilidad[warrior_id, ability_idx, target_pos] = True

        self.max_health_por_tipo = max_health_por_tipo
        self.speed_por_tipo = speed_por_tipo
        self.damage_por_tipo_habilidad = damage_por_tipo_habilidad
        self.turn_cd_por_tipo_habilidad = turn_cd_por_tipo_habilidad
        self.target_mask_por_tipo_habilidad = target_mask_por_tipo_habilidad
        self.effect_type_por_tipo_habilidad = effect_type_por_tipo_habilidad