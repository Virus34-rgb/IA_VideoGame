
import torch

from AI.Environment.abilityData import EffectType


class resolveAction:
    
    def __init__(self,max_health_por_tipo,
               damage_por_tipo_habilidad,
               turn_cd_por_tipo_habilidad,
               target_mask_por_tipo_habilidad,
               effect_type_por_tipo_habilidad,):
        self.max_health_por_tipo = max_health_por_tipo
        self.damage_por_tipo_habilidad = damage_por_tipo_habilidad
        self.turn_cd_por_tipo_habilidad = turn_cd_por_tipo_habilidad
        self.target_mask_por_tipo_habilidad =  target_mask_por_tipo_habilidad
        self.effect_type_por_tipo_habilidad = effect_type_por_tipo_habilidad
        
    def resolve_action(
            self, pos, actors, own_disposition, enemy_disposition, own_health, enemy_health,
            own_cooldowns, own_alive, enemy_alive, actions_actor, enemy_actions,
            own_instance_abilities, enemy_instance_abilities,
            own_castle_slots, enemy_castle_slots,
        ):
            actor_alive_now = own_alive.gather(1, pos.unsqueeze(1)).squeeze(1)
    
            own_slot_abilities = own_instance_abilities.gather(1, pos.view(-1, 1, 1).expand(-1, 1, 4)).squeeze(1)
            ability_pool_idx = own_slot_abilities.gather(1, actions_actor.clamp(0, 3).unsqueeze(1)).squeeze(1)
            effect_type = self.effect_type_por_tipo_habilidad[actors, ability_pool_idx]
    
            es_habilidad = (actions_actor >= 0) & (actions_actor <= 3)
    
            mask_movPos = (actions_actor == 5) & (pos != 2) & actor_alive_now
            mask_movNeg = (actions_actor == 6) & (pos != 0) & actor_alive_now
            mask_self_heal = es_habilidad & (effect_type == EffectType.SELF_HEAL) & actor_alive_now
            mask_team_heal = es_habilidad & (effect_type == EffectType.TEAM_HEAL) & actor_alive_now
            mask_defend = es_habilidad & ((effect_type == EffectType.DEFEND_FULL) | (effect_type == EffectType.DEFEND_HALF)) & actor_alive_now
            mask_ataque = es_habilidad & (effect_type == EffectType.ATTACK) & actor_alive_now
    
            moved, new_disp_mov, new_health_mov, new_cd_mov, new_abilities_mov, new_castle_mov, new_alive_mov = self._resolve_action_movement(
                actors, own_disposition, own_health, own_cooldowns, own_instance_abilities, own_castle_slots, own_alive, actions_actor, pos
            )
    
            own_new_disp = own_disposition.clone()
            own_new_disp = torch.where(mask_movPos.unsqueeze(1), new_disp_mov, own_new_disp)
            own_new_disp = torch.where(mask_movNeg.unsqueeze(1), new_disp_mov, own_new_disp)
    
            own_new_castle = own_castle_slots.clone()
            own_new_castle = torch.where(mask_movPos.unsqueeze(1), new_castle_mov, own_new_castle)
            own_new_castle = torch.where(mask_movNeg.unsqueeze(1), new_castle_mov, own_new_castle)
            
            own_new_alive = own_alive.clone()
            own_new_alive = torch.where(mask_movPos.unsqueeze(1), new_alive_mov, own_new_alive)
            own_new_alive = torch.where(mask_movNeg.unsqueeze(1), new_alive_mov, own_new_alive)
    
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
            wasted_heal = torch.where((mask_self_heal | mask_team_heal) & (heal == 0), 1, 0)
    
            damage_avoided = torch.where(mask_ataque, blocked_raw, torch.zeros_like(blocked_raw))
            
            N = own_disposition.shape[0]
            defense_wasted = torch.zeros(N, dtype=torch.float, device=own_disposition.device)

            if mask_defend.any():
                was_targeted = self._check_if_targeted(
                    pos, enemy_disposition, enemy_actions, enemy_instance_abilities, enemy_alive
                )
                defense_wasted = torch.where(mask_defend & ~was_targeted, 1.0, 0.0)
            return (
                damage, damage_avoided, blocked, moved, heal,
                own_new_disp, enemy_disposition, own_new_health, enemy_new_health,
                own_cd_new, own_new_alive, enemy_alive_final,
                own_abilities_new, ability_pool_idx,
                own_new_castle,wasted_heal,defense_wasted
            )
    
    def _resolve_action_movement(
        self, actors, own_disposition, own_health, own_cooldowns,
        own_instance_abilities, own_castle_slots, own_alive, actions_actor, pos,
    ):
        mask_movPos = (actions_actor == 5) & (pos != 2)
        mask_movNeg = (actions_actor == 6) & (pos != 0)
        moved = (mask_movPos | mask_movNeg).float()

        pos_destino_pos = (pos + 1).clamp(max=2)
        pos_destino_neg = (pos - 1).clamp(min=0)

        # Intercambiar disposición
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

        # Intercambiar salud
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

        # NUEVO: Intercambiar alive (mismo patrón que salud, pero con own_alive)
        own_new_alive = own_alive.clone()
        origen_a = own_alive.gather(1, pos.unsqueeze(1)).squeeze(1)
        destino_a = own_alive.gather(1, pos_destino_pos.unsqueeze(1)).squeeze(1)
        own_new_alive.scatter_(1, pos.unsqueeze(1), torch.where(mask_movPos, destino_a, origen_a).unsqueeze(1))
        own_new_alive.scatter_(1, pos_destino_pos.unsqueeze(1), torch.where(mask_movPos, origen_a, destino_a).unsqueeze(1))

        origen_a2 = own_new_alive.gather(1, pos.unsqueeze(1)).squeeze(1)
        destino_a2 = own_new_alive.gather(1, pos_destino_neg.unsqueeze(1)).squeeze(1)
        own_new_alive_final = own_new_alive.clone()
        own_new_alive_final.scatter_(1, pos.unsqueeze(1), torch.where(mask_movNeg, destino_a2, origen_a2).unsqueeze(1))
        own_new_alive_final.scatter_(1, pos_destino_neg.unsqueeze(1), torch.where(mask_movNeg, origen_a2, destino_a2).unsqueeze(1))

        # Intercambiar cooldowns
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

        # Intercambiar habilidades de instancia
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

        # Intercambiar castle_slots
        own_new_castle = own_castle_slots.clone()
        origen_c = own_castle_slots.gather(1, pos.unsqueeze(1)).squeeze(1)
        destino_c = own_castle_slots.gather(1, pos_destino_pos.unsqueeze(1)).squeeze(1)
        own_new_castle.scatter_(1, pos.unsqueeze(1), torch.where(mask_movPos, destino_c, origen_c).unsqueeze(1))
        own_new_castle.scatter_(1, pos_destino_pos.unsqueeze(1), torch.where(mask_movPos, origen_c, destino_c).unsqueeze(1))

        origen_c2 = own_new_castle.gather(1, pos.unsqueeze(1)).squeeze(1)
        destino_c2 = own_new_castle.gather(1, pos_destino_neg.unsqueeze(1)).squeeze(1)
        own_new_castle_final = own_new_castle.clone()
        own_new_castle_final.scatter_(1, pos.unsqueeze(1), torch.where(mask_movNeg, destino_c2, origen_c2).unsqueeze(1))
        own_new_castle_final.scatter_(1, pos_destino_neg.unsqueeze(1), torch.where(mask_movNeg, origen_c2, destino_c2).unsqueeze(1))

        return (
            moved,
            own_new_disp_final,
            own_new_health_final,
            own_new_cd_final,
            own_new_abilities_final,
            own_new_castle_final,
            own_new_alive_final,   # NUEVO
        )

    def _resolve_action_attack(
        self, actors, ability_pool_idx, enemy_disposition, enemy_health, enemy_alive, enemy_actions,
        enemy_instance_abilities,
    ):
        would_be_damage = self.damage_por_tipo_habilidad[actors, ability_pool_idx]
        target_mask = self.target_mask_por_tipo_habilidad[actors, ability_pool_idx]

        enemy_ability_pool_idx = enemy_instance_abilities.gather(
            2, enemy_actions.clamp(0, 3).unsqueeze(-1)
        ).squeeze(-1)
        enemy_effect_type = self.effect_type_por_tipo_habilidad[enemy_disposition, enemy_ability_pool_idx]
        enemy_es_habilidad = (enemy_actions >= 0) & (enemy_actions <= 3)

        enemy_new_health = enemy_health.clone()
        damage_total = torch.zeros_like(would_be_damage)
        avoided_total = torch.zeros_like(would_be_damage)
        blocks_total = torch.zeros_like(would_be_damage)

        for slot in range(3):
            es_target = target_mask[:, slot] & enemy_alive[:, slot]

            full_block = (enemy_effect_type[:, slot] == EffectType.DEFEND_FULL) & enemy_es_habilidad[:, slot]
            half_block = (enemy_effect_type[:, slot] == EffectType.DEFEND_HALF) & enemy_es_habilidad[:, slot]

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
        max_health_actor = self.max_health_por_tipo[actors]  # (N,)
        current = own_health.gather(1, pos.unsqueeze(1)).squeeze(1)

        valid = max_health_actor > 0
        health_ratio = torch.zeros_like(current)
        health_ratio[valid] = current[valid] / max_health_actor[valid]
        clutch_factor = 1 - health_ratio
        clutch_factor = torch.where(valid, clutch_factor, torch.zeros_like(clutch_factor))

        new_value = torch.min(max_health_actor, current + heal_amount)
        healed = new_value - current
        weighted_healed = healed * clutch_factor

        own_new_health = own_health.scatter(1, pos.unsqueeze(1), new_value.unsqueeze(1))
        return weighted_healed, own_new_health
    
    def _resolve_action_team_heal(self, actors, ability_pool_idx, own_disposition, own_health, own_alive):
        heal_amount = self.damage_por_tipo_habilidad[actors, ability_pool_idx].unsqueeze(1)  # (N, 1)
        max_health_slot = self.max_health_por_tipo[own_disposition]  # (N, 3)

        valid_slot = max_health_slot > 0
        health_ratio = torch.zeros_like(own_health)
        health_ratio[valid_slot] = own_health[valid_slot] / max_health_slot[valid_slot]
        clutch_factor = 1 - health_ratio
        clutch_factor = torch.where(valid_slot, clutch_factor, torch.zeros_like(clutch_factor))

        new_value = torch.min(max_health_slot, own_health + heal_amount)
        healed_per_slot = torch.where(own_alive, new_value - own_health, torch.zeros_like(own_health))
        weighted_healed_per_slot = healed_per_slot * clutch_factor
        total_weighted_healed = weighted_healed_per_slot.sum(dim=1)

        own_new_health = own_health + healed_per_slot
        return total_weighted_healed, own_new_health

    def _update_own_cooldowns(self, actors, accion_actor, ability_pool_idx, pos, own_cooldowns, mask_usa_habilidad):
        turns_cd = self.turn_cd_por_tipo_habilidad[actors, ability_pool_idx]

        slot_expand = pos.view(-1, 1, 1).expand(-1, 1, 4)
        actor_cd = own_cooldowns.gather(1, slot_expand).squeeze(1)
        button_onehot = torch.nn.functional.one_hot(accion_actor.clamp(0, 3), num_classes=4).bool()

        turns_cd_expand = turns_cd.unsqueeze(1).expand(-1, 4)
        marked = torch.where(button_onehot, turns_cd_expand, actor_cd)
        new_actor_cd = torch.where(mask_usa_habilidad.unsqueeze(1), marked, actor_cd)

        return own_cooldowns.scatter(1, slot_expand, new_actor_cd.unsqueeze(1))
    
    def _check_if_targeted(self, pos, enemy_disposition, enemy_actions, enemy_instance_abilities, enemy_alive):
        N = enemy_disposition.shape[0]
        device = enemy_disposition.device
        was_targeted = torch.zeros(N, dtype=torch.bool, device=device)

        # Asegurar pos 1D y tipo long
        pos = pos.view(-1).to(torch.long)
        row_idx = torch.arange(N, device=device)

        for e_slot in range(3):
            alive = enemy_alive[:, e_slot]                 # (N,)
            action = enemy_actions[:, e_slot]              # (N,)
            is_attack_action = (action >= 0) & (action <= 3)
            is_valid = alive & is_attack_action

            # Clamp y asegurar 1D
            action_clamped = action.clamp(min=0, max=3).view(-1)  # (N,)

            # Extraer ability_idx usando gather, más seguro que indexación directa
            abilities = enemy_instance_abilities[:, e_slot]       # (N, 4)
            ability_idx = abilities.gather(1, action_clamped.unsqueeze(1)).squeeze(1)  # (N,)

            enemy_type = enemy_disposition[:, e_slot].view(-1)   # (N,)

            # Indexar tabla de máscaras: resultado debe ser (N, 3)
            target_mask = self.target_mask_por_tipo_habilidad[enemy_type, ability_idx]  # (N, 3)

            # Asegurar forma (N, 3) por si acaso
            if target_mask.dim() == 3:
                if target_mask.shape[1] == 1:
                    target_mask = target_mask.squeeze(1)
                elif target_mask.shape[2] == 1:
                    target_mask = target_mask.squeeze(2)
            elif target_mask.dim() == 1:
                # Caso raro: si devolvió (3,), expandir a (N, 3)
                target_mask = target_mask.unsqueeze(0).expand(N, -1)

            # Obtener el valor de la máscara para la posición del actor
            target_mask_for_pos = target_mask[row_idx, pos]  # (N,)

            # Acumular: si algún enemigo ataca a esta posición
            was_targeted = was_targeted | (is_valid & target_mask_for_pos)

        return was_targeted