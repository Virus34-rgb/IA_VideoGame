"""
Tracking del perfil de estilo del rival (agresividad, movimiento, defensa/cura,
ratio de daño) por partida. Extraído de TrainerV: _compute_profile,
_update_profile_accumulators, _estimate_aggression_prior, _ability_type_masks.

Diseño: los tensores de perfil (_p1_profile, _p2_profile) viven aquí como
self.X -- son el estado que esta clase existe para mantener. TrainerV los lee
via propiedad para pasarlos a ObservationV/logging, pero nunca los escribe
directamente (evita la tentación de "actualizar el perfil desde fuera").
"""
import torch

from AI.Environment.abilityData import EffectType
import constants


class OpponentProfileTracker:
    def __init__(self, N: int, environment):
        self.N = N
        self.environment = environment
        self._p1_profile = torch.full((N, 4), 0.5)
        self._p2_profile = torch.full((N, 4), 0.5)

    @property
    def p1_profile(self) -> torch.Tensor:
        return self._p1_profile

    @property
    def p2_profile(self) -> torch.Tensor:
        return self._p2_profile

    def reset(self):
        self._p1_profile = torch.full((self.N, 4), 0.5)
        self._p2_profile = torch.full((self.N, 4), 0.5)

    def set_initial_priors(self, p1_disposition, p1_instance_abilities, p2_disposition, p2_instance_abilities):
        self._p1_profile[:, 0] = self._estimate_aggression_prior(p1_disposition, p1_instance_abilities)
        self._p2_profile[:, 0] = self._estimate_aggression_prior(p2_disposition, p2_instance_abilities)

    def _estimate_aggression_prior(self, disposition, instance_abilities):
        damage = self.environment.damage_por_tipo_habilidad[disposition.unsqueeze(-1), instance_abilities]
        effect_type = self.environment.effect_type_por_tipo_habilidad[disposition.unsqueeze(-1), instance_abilities]
        attack_mask = effect_type == EffectType.ATTACK
        attack_damage = damage * attack_mask
        total_damage = attack_damage.sum(dim=(1, 2))
        prior = total_damage / constants.PROFILE_DAMAGE_POTENTIAL_REF
        return prior.clamp(0.0, 1.0)

    def _ability_type_masks(self, disposition, instance_abilities):
        tipo_expand = disposition.unsqueeze(-1).expand(-1, -1, 4)
        effect_type = self.environment.effect_type_por_tipo_habilidad[tipo_expand, instance_abilities]
        es_ataque = effect_type == EffectType.ATTACK
        es_defensa_cura = (
            (effect_type == EffectType.DEFEND_FULL) | (effect_type == EffectType.DEFEND_HALF) |
            (effect_type == EffectType.SELF_HEAL) | (effect_type == EffectType.TEAM_HEAL)
        )
        return es_ataque, es_defensa_cura

    def update_after_turn(
        self, p1_types_now, p1_abilities_now, p1_action_mask_now, p1_alive_now,
        p2_types_now, p2_abilities_now, p2_action_mask_now, p2_alive_now,
    ):
        p1_alive_count = p1_alive_now.sum(dim=-1).float().clamp(min=1.0)
        p1_atk_taken = self.environment.p1_attacks.float()
        p1_def_taken = self.environment.p1_defenses.float()
        p1_move_taken = self.environment.p1_movements.float()
        p1_dmg_dealt = self.environment.p1_damage
        p1_dmg_recv = self.environment.p2_damage
            
        p2_alive_count = p2_alive_now.sum(dim=-1).float().clamp(min=1.0)
        p2_atk_taken = self.environment.p2_attacks.float()
        p2_def_taken = self.environment.p2_defenses.float()
        p2_move_taken = self.environment.p2_movements.float()
        p2_dmg_dealt = self.environment.p2_damage
        p2_dmg_recv = self.environment.p1_damage

        self._p1_profile = self._compute_profile(
            self._p1_profile, p1_atk_taken, p1_alive_count, p1_def_taken, p1_alive_count,
            p1_move_taken, p1_alive_count, p1_dmg_dealt, p1_dmg_recv,
        )
        self._p2_profile = self._compute_profile(
            self._p2_profile, p2_atk_taken, p2_alive_count, p2_def_taken, p2_alive_count,
            p2_move_taken, p2_alive_count, p2_dmg_dealt, p2_dmg_recv,
        )

    def _compute_profile(self, old_profile, atk_taken, atk_opp, def_taken, def_opp, move_taken, move_opp, dmg_dealt, dmg_recv):
        decay = constants.PROFILE_EMA_DECAY
        
        new_aggression = (atk_taken / atk_opp).clamp(0.0, 1.0)
        new_movement   = (move_taken / move_opp).clamp(0.0, 1.0)
        new_defense    = (def_taken / def_opp).clamp(0.0, 1.0)

        aggression = (decay * new_aggression + (1 - decay) * old_profile[:, 0]).clamp(0.0, 1.0)
        movement_freq = (decay * new_movement + (1 - decay) * old_profile[:, 1]).clamp(0.0, 1.0)
        defense_usage = (decay * new_defense + (1 - decay) * old_profile[:, 2]).clamp(0.0, 1.0)

        damage_ratio_raw = dmg_dealt / dmg_recv.clamp(min=1)
        new_damage_ratio = damage_ratio_raw / (damage_ratio_raw + 1.0)   # ya acotado a (0,1) por construcción matemática
        damage_ratio = (decay * new_damage_ratio + (1 - decay) * old_profile[:, 3]).clamp(0.0, 1.0)

        return torch.stack([aggression, movement_freq, defense_usage, damage_ratio], dim=-1)