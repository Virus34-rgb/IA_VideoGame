
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
    
            moved, new_disp_mov, new_health_mov, new_cd_mov, new_abilities_mov, new_castle_mov, new_alive_mov,strategic_movement = self._resolve_action_movement(
                actors, own_disposition, own_health, own_cooldowns, own_instance_abilities, 
                own_castle_slots, own_alive, actions_actor, pos,
                enemy_disposition, enemy_alive, enemy_actions, enemy_instance_abilities,
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
    
            damage_raw, blocked_raw, enemy_health_after_attack, enemy_alive_after_attack, overkill_damage, kill_confirmed, = self._resolve_action_attack(
                actors, ability_pool_idx, enemy_disposition, enemy_health, enemy_alive, enemy_actions, enemy_instance_abilities,
            )
            damage = damage_raw * mask_ataque.float()
            blocked = blocked_raw * mask_ataque.float()
            overkill_damage = overkill_damage * mask_ataque.float()
            kill_confirmed = kill_confirmed * mask_ataque.float()
    
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
    
            heal = healed_self * mask_self_heal.float() + healed_team * mask_team_heal.float()
            wasted_heal = ((mask_self_heal | mask_team_heal) & (heal == 0)).float()
    
            damage_avoided = torch.where(mask_ataque, blocked_raw, torch.zeros_like(blocked_raw))
            
            N = own_disposition.shape[0]
            defense_wasted = torch.zeros(N, dtype=torch.float, device=own_disposition.device)

            if mask_defend.any():
                was_targeted = self._check_if_targeted(
                    pos, enemy_disposition, enemy_actions, enemy_instance_abilities, enemy_alive
                )
                defense_wasted = (mask_defend & ~was_targeted).float()
                
            mask_cura = mask_self_heal | mask_team_heal
            
            return (
                damage, damage_avoided, blocked, moved, heal,
                own_new_disp, enemy_disposition, own_new_health, enemy_new_health,
                own_cd_new, own_new_alive, enemy_alive_final,
                own_abilities_new, ability_pool_idx,
                own_new_castle,wasted_heal,defense_wasted,strategic_movement,overkill_damage,kill_confirmed,
                mask_ataque, mask_defend, mask_cura
            )
    
    def _swap_by_position(self, tensor, pos_a, pos_b, mask):
        """
        Intercambia, para las filas donde mask es True, los valores del tensor
        entre la posición pos_a y pos_b (a lo largo de dim=1). Para las filas
        donde mask es False, el tensor no cambia.
        Soporta tensores (N, 3) y (N, 3, K) (K=4 para cooldowns/habilidades).
        pos_a, pos_b: (N,) long. mask: (N,) bool (se expande internamente).
        """
        extra_dims = tensor.dim() - 2   # 0 para (N,3), 1 para (N,3,4)
        pos_a_idx = pos_a.view(-1, *([1] * (extra_dims + 1)))
        pos_b_idx = pos_b.view(-1, *([1] * (extra_dims + 1)))
        if extra_dims == 1:
            K = tensor.shape[-1]
            pos_a_idx = pos_a_idx.expand(-1, 1, K)
            pos_b_idx = pos_b_idx.expand(-1, 1, K)
            mask_exp = mask.view(-1, 1, 1).expand(-1, 1, K)
        else:
            mask_exp = mask.view(-1, 1)

        val_a = tensor.gather(1, pos_a_idx)
        val_b = tensor.gather(1, pos_b_idx)

        result = tensor.clone()
        result.scatter_(1, pos_a_idx, torch.where(mask_exp, val_b, val_a))
        result.scatter_(1, pos_b_idx, torch.where(mask_exp, val_a, val_b))
        return result

    def _resolve_action_movement(
        self, actors, own_disposition, own_health, own_cooldowns,
        own_instance_abilities, own_castle_slots, own_alive, actions_actor, pos,
        enemy_disposition, enemy_alive, enemy_actions, enemy_instance_abilities,
    ):
        mask_movPos = (actions_actor == 5) & (pos != 2)
        mask_movNeg = (actions_actor == 6) & (pos != 0)
        moved = (mask_movPos | mask_movNeg).float()
        mask_move = mask_movPos | mask_movNeg   # excluyentes entre sí

        # Destino único: pos+1 si es movPos, pos-1 si es movNeg, pos si no se mueve
        # (para las filas que no se mueven, destino==pos y el swap es un no-op).
        pos_destino = torch.where(
            mask_movPos, (pos + 1).clamp(max=2),
            torch.where(mask_movNeg, (pos - 1).clamp(min=0), pos),
        )

        new_pos = pos_destino.clamp(0, 2)

        # Calcular targeted antes y después
        targeted_by_enemy = self._check_if_targeted(
            pos, enemy_disposition, enemy_actions, enemy_instance_abilities, enemy_alive
        )
        targeted_by_enemy_post = self._check_if_targeted(
            new_pos, enemy_disposition, enemy_actions, enemy_instance_abilities, enemy_alive
        )
        strategic_movement = mask_move & targeted_by_enemy & ~targeted_by_enemy_post

        # Un único swap por campo (mask_move ya cubre pos+ y pos- combinados,
        # porque pos_destino ya codifica la dirección correcta por fila).
        own_new_disp = self._swap_by_position(own_disposition, pos, pos_destino, mask_move)
        own_new_health = self._swap_by_position(own_health, pos, pos_destino, mask_move)
        own_new_alive = self._swap_by_position(own_alive, pos, pos_destino, mask_move)
        own_new_cd = self._swap_by_position(own_cooldowns, pos, pos_destino, mask_move)
        own_new_abilities = self._swap_by_position(own_instance_abilities, pos, pos_destino, mask_move)
        own_new_castle = self._swap_by_position(own_castle_slots, pos, pos_destino, mask_move)

        return (
            moved,
            own_new_disp,
            own_new_health,
            own_new_cd,
            own_new_abilities,
            own_new_castle,
            own_new_alive,
            strategic_movement,
        )

    def _resolve_action_attack(
        self, actors, ability_pool_idx, enemy_disposition, enemy_health, enemy_alive, enemy_actions,
        enemy_instance_abilities,
    ):
        would_be_damage = self.damage_por_tipo_habilidad[actors, ability_pool_idx]        # (N,)
        target_mask = self.target_mask_por_tipo_habilidad[actors, ability_pool_idx]        # (N,3)

        enemy_ability_pool_idx = enemy_instance_abilities.gather(
            2, enemy_actions.clamp(0, 3).unsqueeze(-1)
        ).squeeze(-1)                                                                      # (N,3)
        enemy_effect_type = self.effect_type_por_tipo_habilidad[enemy_disposition, enemy_ability_pool_idx]  # (N,3)
        enemy_es_habilidad = (enemy_actions >= 0) & (enemy_actions <= 3)                   # (N,3)

        es_target = target_mask & enemy_alive[:, :3]                                       # (N,3)

        full_block = (enemy_effect_type == EffectType.DEFEND_FULL) & enemy_es_habilidad     # (N,3)
        half_block = (enemy_effect_type == EffectType.DEFEND_HALF) & enemy_es_habilidad     # (N,3)

        dmg = would_be_damage.unsqueeze(1)     # (N,1) -> broadcast contra (N,3)
        zero = torch.zeros_like(dmg)           # (N,1)

        hit_damage = torch.where(full_block, zero, torch.where(half_block, dmg / 2, dmg))   # (N,3)
        avoided = torch.where(full_block, dmg, torch.where(half_block, dmg / 2, zero))       # (N,3)
        blocked_flag = (full_block | half_block).float()                                    # (N,3)

        hit_damage = torch.where(es_target, hit_damage, torch.zeros_like(hit_damage))
        avoided = torch.where(es_target, avoided, torch.zeros_like(avoided))
        blocked_flag = torch.where(es_target, blocked_flag, torch.zeros_like(blocked_flag))

        health_before = enemy_health                                                        # (N,3)
        lethal = es_target & (health_before > 0) & (hit_damage >= health_before)             # (N,3)
        overkill_per_slot = torch.where(lethal, hit_damage - health_before, torch.zeros_like(hit_damage))
        kill_per_slot = lethal.float()

        enemy_new_health = torch.where(es_target, health_before - hit_damage, health_before)  # (N,3)
        enemy_new_alive = enemy_alive & (enemy_new_health > 0)

        damage_total = hit_damage.sum(dim=1)
        avoided_total = avoided.sum(dim=1)
        blocks_total = blocked_flag.sum(dim=1)
        overkill_damage = overkill_per_slot.sum(dim=1)
        kills_this_action = kill_per_slot.sum(dim=1)

        return damage_total, blocks_total, enemy_new_health, enemy_new_alive, overkill_damage, kills_this_action

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

        pos = pos.view(-1).to(torch.long)   # (N,)
        N = enemy_disposition.shape[0]

        alive = enemy_alive                                                    # (N,3)
        action = enemy_actions                                                 # (N,3)
        is_attack_action = (action >= 0) & (action <= 3)
        is_valid = alive & is_attack_action                                    # (N,3)

        action_clamped = action.clamp(min=0, max=3)                            # (N,3)
        ability_idx = enemy_instance_abilities.gather(2, action_clamped.unsqueeze(-1)).squeeze(-1)  # (N,3)

        enemy_type = enemy_disposition                                         # (N,3)

        # (num_types, POOL, 3) indexado con (N,3) y (N,3) -> (N,3,3)
        # dim1 = slot enemigo que ataca, dim2 = posición objetivo de esa habilidad
        target_mask = self.target_mask_por_tipo_habilidad[enemy_type, ability_idx]  # (N,3,3)

        pos_exp = pos.view(N, 1, 1).expand(N, 3, 1)
        target_mask_for_pos = target_mask.gather(2, pos_exp).squeeze(-1)       # (N,3)

        was_targeted_per_enemy_slot = is_valid & target_mask_for_pos           # (N,3)
        return was_targeted_per_enemy_slot.any(dim=1)                          # (N,)