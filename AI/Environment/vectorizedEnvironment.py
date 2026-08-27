import torch

from AI.Environment.gameState import GameState
from AI.Environment.statsV import StatsV
from AI.Environment.warriorFactory import get_warriors_classes
from constants import DISCOUNT_FACTOR, REWARD_WEIGHTS, TURN_PENALTY, WIN_REWARD, MAX_TURNS, MAX_DEATHS_PER_TEAM


class VectorizedEnvironment:
    def __init__(self, N):
        self.N = N
        self.indices = torch.arange(N)
        self.warriors_classes = get_warriors_classes()
        self.p1_disposition = torch.zeros((N, 3), dtype=torch.long)
        self.p2_disposition = torch.zeros((N, 3), dtype=torch.long)
        self.p1_healths = torch.zeros((N, 3), dtype=torch.float)
        self.p2_healths = torch.zeros((N, 3), dtype=torch.float)
        self.p1_cooldowns = torch.zeros((N, 3, 4), dtype=torch.bool)
        self.p2_cooldowns = torch.zeros((N, 3, 4), dtype=torch.bool)
        self.p1_alive = torch.zeros((N, 3), dtype=torch.bool)
        self.p2_alive = torch.zeros((N, 3), dtype=torch.bool)
        self.p1_initialWarrior = torch.zeros(N, dtype=torch.long)
        self.p2_initialWarrior = torch.zeros(N, dtype=torch.long)
        self.p1_initialPosition = torch.zeros(N, dtype=torch.long)
        self.p2_initialPosition = torch.zeros(N, dtype=torch.long)
        self.p1_deaths = torch.zeros(N, dtype=torch.long)
        self.p2_deaths = torch.zeros(N, dtype=torch.long)
        self.ended = torch.zeros(N, dtype=torch.bool)
        self.winner = torch.full((N,), -1, dtype=torch.long)
        self.turn_number = torch.zeros(N, dtype=torch.int)
        self._build_static_tables()
        self.stats = StatsV()

    def reset(self):
        self.p1_disposition = torch.zeros((self.N, 3), dtype=torch.long)
        self.p2_disposition = torch.zeros((self.N, 3), dtype=torch.long)
        self.p1_healths = torch.zeros((self.N, 3), dtype=torch.float)
        self.p2_healths = torch.zeros((self.N, 3), dtype=torch.float)
        self.p1_cooldowns = torch.zeros((self.N, 3, 4), dtype=torch.bool)
        self.p2_cooldowns = torch.zeros((self.N, 3, 4), dtype=torch.bool)
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
        self.stats.start_batch(self.N)
        return self.get_state()

    def get_state(self):
        return GameState(self.p1_disposition, self.p2_disposition,
                          self.p1_deaths, self.p2_deaths, self.turn_number)

    def team_selection(self, warrior_p1, pos1, warrior_p2, pos2, selected, health1, health2):
        self.warrior_selected(warrior_p1, pos1, warrior_p2, pos2, selected, health1, health2)
        return self.get_state()

    def warrior_selected(self, warrior1, pos1, warrior2, pos2, selected, health1, health2):
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
        self.stats.accumulate_warrior_use(warrior1, warrior2)

    def turn(self, actionsp1, actionsp2):
        self.turn_number += 1
        ya_terminadas_antes = self.ended.clone()

        order, actor_alive_inicio = self.get_turn_order()
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

        p1_health = (self.p1_healths / self.max_health_por_tipo[self.p1_disposition]).sum(dim=1)
        p2_health = (self.p2_healths / self.max_health_por_tipo[self.p2_disposition]).sum(dim=1)

        for position in range(6):
            actor_idx = order[:, position]
            player = actor_idx // 3
            pos = actor_idx % 3
            es_p1 = (player == 0)                  
            player_mask = es_p1.unsqueeze(1)
            player_mask_3 = player_mask.unsqueeze(-1)

            own_disposition = torch.where(player_mask, self.p1_disposition, self.p2_disposition)
            enemy_disposition = torch.where(player_mask, self.p2_disposition, self.p1_disposition)
            own_health = torch.where(player_mask, self.p1_healths, self.p2_healths)
            enemy_health = torch.where(player_mask, self.p2_healths, self.p1_healths)
            own_actions = torch.where(player_mask, actionsp1, actionsp2)
            accion_actor = own_actions.gather(1, pos.unsqueeze(1)).squeeze(1)
            enemy_actions = torch.where(player_mask, actionsp2, actionsp1)
            own_alive = torch.where(player_mask, self.p1_alive, self.p2_alive)
            enemy_alive = torch.where(player_mask, self.p2_alive, self.p1_alive)
            own_cooldowns = torch.where(player_mask_3, self.p1_cooldowns, self.p2_cooldowns)
            tipo_actor = own_disposition.gather(1, pos.unsqueeze(1)).squeeze(1)

            (dmg, avoided, blocks, moved, heal, own_new_disp, enemy_new_disp, own_new_health,
            enemy_new_health, own_cd_new, own_alive_new, enemy_alive_new
            ) = self._resolve_action(pos, tipo_actor, own_disposition, enemy_disposition, own_health,
                                    enemy_health, own_cooldowns, own_alive, enemy_alive,
                                    accion_actor, enemy_actions)

            self.p1_disposition = torch.where(player_mask, own_new_disp, enemy_new_disp)
            self.p2_disposition = torch.where(player_mask, enemy_new_disp, own_new_disp)
            self.p1_healths = torch.where(player_mask, own_new_health, enemy_new_health)
            self.p2_healths = torch.where(player_mask, enemy_new_health, own_new_health)
            self.p1_alive = torch.where(player_mask, own_alive_new, enemy_alive_new)
            self.p2_alive = torch.where(player_mask, enemy_alive_new, own_alive_new)
            self.p1_cooldowns = torch.where(player_mask_3, own_cd_new, self.p1_cooldowns)
            self.p2_cooldowns = torch.where(~player_mask_3, own_cd_new, self.p2_cooldowns)

            damage_p1 += torch.where(es_p1, dmg, torch.zeros_like(dmg))
            damage_p2 += torch.where(es_p1, torch.zeros_like(dmg), dmg)
            damage_avoided_p1 += torch.where(~es_p1, avoided, torch.zeros_like(avoided))
            damage_avoided_p2 += torch.where(~es_p1, torch.zeros_like(avoided), avoided)
            blocks_p1 += torch.where(~es_p1, blocks, torch.zeros_like(blocks))
            blocks_p2 += torch.where(~es_p1, torch.zeros_like(blocks), blocks)
            heal_p1 += torch.where(es_p1, heal, torch.zeros_like(heal))
            heal_p2 += torch.where(~es_p1, heal, torch.zeros_like(heal))

            self.stats.accumulate_movements(moved, es_p1, ~ya_terminadas_antes)
            self.stats.accumulate_attacks(tipo_actor, accion_actor, es_p1, ~ya_terminadas_antes)

        p1_post_health = (self.p1_healths / self.max_health_por_tipo[self.p1_disposition]).sum(dim=1)
        p2_post_health = (self.p2_healths / self.max_health_por_tipo[self.p2_disposition]).sum(dim=1)
        shapping1 = p1_health / 3 - p2_health / 3
        shapping2 = p1_post_health / 3 - p2_post_health / 3

        p1_new_deaths = (p1_alive_inicio & ~self.p1_alive).sum(dim=1).to(self.p1_deaths.dtype)
        p2_new_deaths = (p2_alive_inicio & ~self.p2_alive).sum(dim=1).to(self.p2_deaths.dtype)

        self.stats.accumulate_turn(damage_p1, damage_p2, blocks_p1, blocks_p2,
                                    damage_avoided_p1, damage_avoided_p2, heal_p1, heal_p2,
                                    ya_terminadas_antes)

        rewardP1, rewardP2 = self.calculate_rewards(
            damage_p1, damage_p2, damage_avoided_p1, damage_avoided_p2, heal_p1, heal_p2,
            shapping1, shapping2, p1_new_deaths, p2_new_deaths
        )

        rewardP1 = torch.where(ya_terminadas_antes, torch.zeros_like(rewardP1), rewardP1)
        rewardP2 = torch.where(ya_terminadas_antes, torch.zeros_like(rewardP2), rewardP2)

        return self.get_state(), rewardP1, rewardP2, self.ended
    
    def _resolve_action(self,pos,actors,own_disposition,enemy_disposition,own_health,enemy_health,
                        own_cooldowns,own_alive, enemy_alive,actions_actor, enemy_actions):

        mask_movPos = (actions_actor == 5) & (pos != 2)
        mask_movNeg = (actions_actor == 6) & (pos != 0)
        mask_self_heal = (actors == 2) & (actions_actor == 1)
        mask_team_heal = (actors == 5) & (actions_actor == 1)
        mask_defend = (((actors == 5) & (actions_actor == 3)) |((actors == 3) & (actions_actor == 2)) |
                    ((actors == 1) & (actions_actor == 2)))
        mask_ataque = ~(mask_movPos | mask_movNeg | mask_self_heal | mask_team_heal | mask_defend)

        mask_movPos_3 = mask_movPos.unsqueeze(1)
        mask_movNeg_3 = mask_movNeg.unsqueeze(1)
        mask_self_heal_3 = mask_self_heal.unsqueeze(1)
        mask_team_heal_3 = mask_team_heal.unsqueeze(1)
        mask_ataque_3 = mask_ataque.unsqueeze(1)
        mask_movPos_4 = mask_movPos.view(-1, 1, 1)
        mask_movNeg_4 = mask_movNeg.view(-1, 1, 1)

        (moved, own_new_disposition_movement,
         own_new_health_movement, own_new_cd_movement) = self._resolve_action_movement(
            actors, own_disposition, own_health, own_cooldowns, actions_actor, pos
        )

        own_new_disp = own_disposition.clone()
        own_new_disp = torch.where(mask_movNeg_3,own_new_disposition_movement,own_new_disp)
        own_new_disp = torch.where(mask_movPos_3,own_new_disposition_movement,own_new_disp)

        (damage_raw,blocked_raw,enemy_health_atacado,enemy_new_alive) = self._resolve_action_attack(actors,actions_actor,
                                                                                                    enemy_disposition,
                                                                                                    enemy_health,
                                                                                                    enemy_alive,
                                                                                                    enemy_actions)

        # El daño solo se aplica si realmente era una acción de ataque.
        damage = torch.where(mask_ataque,damage_raw,torch.zeros_like(damage_raw))

        blocked = torch.where(mask_ataque,blocked_raw,torch.zeros_like(blocked_raw))

        healed_self, own_health_self = self._resolve_action_self_heal(actors,actions_actor,pos,own_health)
        healed_team, own_health_team = self._resolve_action_team_heal(actors,actions_actor,own_disposition,
                                                                      own_health,own_alive)

        own_new_health = own_health.clone()
        own_new_health = torch.where(mask_movNeg_3, own_new_health_movement, own_new_health)
        own_new_health = torch.where(mask_movPos_3, own_new_health_movement, own_new_health)
        own_new_health = torch.where(mask_self_heal_3,own_health_self,own_new_health)
        own_new_health = torch.where(mask_team_heal_3,own_health_team,own_new_health)
        enemy_new_health = torch.where(mask_ataque_3,enemy_health_atacado,enemy_health)
        
        mask_usa_habilidad = (mask_ataque | mask_self_heal | mask_team_heal | mask_defend)

        own_cd_new = self._update_own_cooldowns(actors,actions_actor,pos,own_cooldowns,mask_usa_habilidad)

        own_cd_new = torch.where(mask_movNeg_4, own_new_cd_movement, own_cd_new)
        own_cd_new = torch.where(mask_movPos_4, own_new_cd_movement, own_cd_new)

        enemy_alive_final = torch.where(mask_ataque_3,enemy_new_alive,enemy_alive)
        heal = torch.zeros_like(own_health[:, 0])
        heal = torch.where(mask_self_heal,healed_self,heal)
        heal = torch.where(mask_team_heal,healed_team,heal)
        damage_avoided = torch.where(mask_ataque,blocked_raw,torch.zeros_like(blocked_raw))

        return (damage,damage_avoided,blocked,moved,heal,
            own_new_disp,enemy_disposition,own_new_health,enemy_new_health,
            own_cd_new,own_alive,enemy_alive_final)

    def _resolve_action_movement(self, actors, own_disposition, own_health, own_cooldowns, actions_actor, pos):
        mask_movPos = (actions_actor == 5) & (pos != 2)
        mask_movNeg = (actions_actor == 6) & (pos != 0)
        moved = (mask_movPos | mask_movNeg).float()

        pos_destino_pos = (pos + 1).clamp(max=2)
        pos_destino_neg = (pos - 1).clamp(min=0)

        own_new_disposition = own_disposition.clone()
        origen = own_disposition.gather(1, pos.unsqueeze(1)).squeeze(1)
        destino = own_disposition.gather(1, pos_destino_pos.unsqueeze(1)).squeeze(1)
        own_new_disposition.scatter_(1, pos.unsqueeze(1), torch.where(mask_movPos, destino, origen).unsqueeze(1))
        own_new_disposition.scatter_(1, pos_destino_pos.unsqueeze(1), torch.where(mask_movPos, origen, destino).unsqueeze(1))

        origen2 = own_new_disposition.gather(1, pos.unsqueeze(1)).squeeze(1)
        destino2 = own_new_disposition.gather(1, pos_destino_neg.unsqueeze(1)).squeeze(1)
        own_new_disposition_final = own_new_disposition.clone()
        own_new_disposition_final.scatter_(1, pos.unsqueeze(1), torch.where(mask_movNeg, destino2, origen2).unsqueeze(1))
        own_new_disposition_final.scatter_(1, pos_destino_neg.unsqueeze(1), torch.where(mask_movNeg, origen2, destino2).unsqueeze(1))

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

        return moved, own_new_disposition_final, own_new_health_final, own_new_cd_final

    def _resolve_action_attack(self, actors, accion_actor, enemy_disposition, enemy_health, enemy_alive, enemy_actions):
        ability_idx = accion_actor.clamp(0, 3)
        would_be_damage = self.damage_por_tipo_habilidad[actors, ability_idx]
        target_mask = self.target_mask_por_tipo_habilidad[actors, ability_idx]

        enemy_new_health = enemy_health.clone()
        damage_total = torch.zeros_like(would_be_damage)
        avoided_total = torch.zeros_like(would_be_damage)
        blocks_total = torch.zeros_like(would_be_damage)

        for slot in range(3):
            es_target = target_mask[:, slot] & enemy_alive[:, slot]
            enemy_id_slot = enemy_disposition[:, slot]
            enemy_action_slot = enemy_actions[:, slot]

            full_block = ((enemy_id_slot == 1) | (enemy_id_slot == 3)) & (enemy_action_slot == 2)
            half_block = (enemy_id_slot == 5) & (enemy_action_slot == 3)

            hit_damage = torch.where(full_block, torch.zeros_like(would_be_damage),
                                      torch.where(half_block, would_be_damage / 2, would_be_damage))
            avoided = torch.where(full_block, would_be_damage,
                                   torch.where(half_block, would_be_damage / 2, torch.zeros_like(would_be_damage)))
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

    def _resolve_action_self_heal(self, actors, accion_actor, pos, own_health):
        ability_idx = accion_actor.clamp(0, 3)
        heal_amount = self.damage_por_tipo_habilidad[actors, ability_idx]
        max_health_actor = self.max_health_por_tipo[actors]
        current = own_health.gather(1, pos.unsqueeze(1)).squeeze(1)
        nuevo_valor = torch.min(max_health_actor, current + heal_amount)
        healed = nuevo_valor - current
        own_new_health = own_health.scatter(1, pos.unsqueeze(1), nuevo_valor.unsqueeze(1))
        return healed, own_new_health

    def _resolve_action_team_heal(self, actors, accion_actor, own_disposition, own_health, own_alive):
        ability_idx = accion_actor.clamp(0, 3)
        heal_amount = self.damage_por_tipo_habilidad[actors, ability_idx].unsqueeze(1)
        max_health_por_slot = self.max_health_por_tipo[own_disposition]
        nuevo_valor = torch.min(max_health_por_slot, own_health + heal_amount)
        healed_por_slot = torch.where(own_alive, nuevo_valor - own_health, torch.zeros_like(own_health))
        own_new_health = own_health + healed_por_slot
        total_healed = healed_por_slot.sum(dim=1)
        return total_healed, own_new_health

    def _update_own_cooldowns(self, actors, accion_actor, pos, own_cooldowns, mask_usa_habilidad):
        ability_idx = accion_actor.clamp(0, 3)
        can_repeat = self.can_repeat_por_tipo_habilidad[actors, ability_idx]
        slot_expand = pos.view(-1, 1, 1).expand(-1, 1, 4)
        actor_cd = own_cooldowns.gather(1, slot_expand).squeeze(1)
        reset_cd = torch.zeros_like(actor_cd)
        ability_onehot = torch.nn.functional.one_hot(ability_idx, num_classes=4).bool()
        marcado = torch.where((~can_repeat).unsqueeze(1), reset_cd | ability_onehot, reset_cd)
        nuevo_actor_cd = torch.where(mask_usa_habilidad.unsqueeze(1), marcado, actor_cd)
        return own_cooldowns.scatter(1, slot_expand, nuevo_actor_cd.unsqueeze(1))

    def get_turn_order(self):
        actor_types = torch.cat([self.p1_disposition, self.p2_disposition], dim=1)
        actor_alive = torch.cat([self.p1_alive, self.p2_alive], dim=1)
        speeds = self.speed_por_tipo[actor_types]
        speeds = torch.where(actor_alive, speeds, torch.full_like(speeds, float('-inf')))
        order = torch.argsort(speeds, dim=1, descending=True, stable=True)
        return order, actor_alive

    def _check_end_conditions(self):
        ya_terminadas = self.ended.clone()

        ambos = (self.p1_deaths >= MAX_DEATHS_PER_TEAM) & (self.p2_deaths >= MAX_DEATHS_PER_TEAM)
        solo_p1 = (self.p1_deaths >= MAX_DEATHS_PER_TEAM) & ~ambos
        solo_p2 = (self.p2_deaths >= MAX_DEATHS_PER_TEAM) & ~ambos & ~solo_p1
        por_turnos = (self.turn_number > MAX_TURNS) & ~(ambos | solo_p1 | solo_p2)

        nuevo_winner = self.winner.clone()
        nuevo_winner = torch.where(ambos, torch.full_like(nuevo_winner, 2), nuevo_winner)
        nuevo_winner = torch.where(solo_p1, torch.full_like(nuevo_winner, 1), nuevo_winner)
        nuevo_winner = torch.where(solo_p2, torch.full_like(nuevo_winner, 0), nuevo_winner)
        nuevo_winner = torch.where(por_turnos, torch.full_like(nuevo_winner, 2), nuevo_winner)

        termina_ahora = (ambos | solo_p1 | solo_p2 | por_turnos) & ~ya_terminadas

        self.winner = torch.where(termina_ahora, nuevo_winner, self.winner)
        self.ended = self.ended | termina_ahora

        self.stats.close_finished_games(
            termina_ahora, self.winner, self.p1_deaths, self.p2_deaths, self.turn_number,
            por_muerte_mask=(ambos | solo_p1 | solo_p2), por_turnos_mask=por_turnos
        )

    def _reward(self, **components):
        weighted = sum(REWARD_WEIGHTS[name] * value for name, value in components.items())
        return weighted - TURN_PENALTY

    def calculate_rewards(self, damage_p1, damage_p2, damage_avoided_p1, damage_avoided_p2,
                           healed_p1, healed_p2, shapping1, shapping2, newDeaths_p1, newDeaths_p2):
        self.p1_deaths += newDeaths_p1
        self.p2_deaths += newDeaths_p2
        self._check_end_conditions()

        win_p1 = torch.where(
            self.winner == 0, torch.full_like(damage_p1, WIN_REWARD),
            torch.where(self.winner == 1, torch.full_like(damage_p1, -WIN_REWARD), torch.zeros_like(damage_p1))
        )
        win_p2 = -win_p1

        rewardP1 = self._reward(damage=damage_p1 - damage_p2, deaths=newDeaths_p1 - newDeaths_p2,
                                 win=win_p1, blocks=damage_avoided_p1, heal=healed_p1,
                                 shaping_weight=DISCOUNT_FACTOR * shapping2 - shapping1)
        rewardP2 = self._reward(damage=damage_p2 - damage_p1, deaths=newDeaths_p2 - newDeaths_p1,
                                 win=win_p2, blocks=damage_avoided_p2, heal=healed_p2,
                                 shaping_weight=-(DISCOUNT_FACTOR * shapping2 - shapping1))
        return rewardP1, rewardP2

    def _build_static_tables(self):
        num_types = max(self.warriors_classes.keys()) + 1
        num_abilities = 4
        num_slots = 3
        max_health_por_tipo = torch.zeros(num_types, dtype=torch.float)
        speed_por_tipo = torch.zeros(num_types, dtype=torch.float)
        damage_por_tipo_habilidad = torch.zeros(num_types, num_abilities, dtype=torch.float)
        can_repeat_por_tipo_habilidad = torch.zeros(num_types, num_abilities, dtype=torch.bool)
        target_mask_por_tipo_habilidad = torch.zeros(num_types, num_abilities, num_slots, dtype=torch.bool)

        for warrior_id, warrior_data in self.warriors_classes.items():
            max_health_por_tipo[warrior_id] = warrior_data.max_health
            speed_por_tipo[warrior_id] = warrior_data.speed
            for ability_idx, ability in enumerate(warrior_data.abilities):
                damage_por_tipo_habilidad[warrior_id, ability_idx] = ability.damage
                can_repeat_por_tipo_habilidad[warrior_id, ability_idx] = ability.can_repeat
                for target_pos in ability.target_positions:
                    target_mask_por_tipo_habilidad[warrior_id, ability_idx, target_pos] = True

        self.max_health_por_tipo = max_health_por_tipo
        self.speed_por_tipo = speed_por_tipo
        self.damage_por_tipo_habilidad = damage_por_tipo_habilidad
        self.can_repeat_por_tipo_habilidad = can_repeat_por_tipo_habilidad
        self.target_mask_por_tipo_habilidad = target_mask_por_tipo_habilidad