"""
Test de equivalencia funcional para las versiones vectorizadas de
_resolve_action_attack y _check_if_targeted frente a implementaciones de
referencia con bucle Python explícito (idénticas a las originales del proyecto).
"""
import torch
from AI.Environment.resolve_actions import resolveAction
from AI.Environment.abilityData import EffectType


def _make_resolver():
    num_types, num_abilities, num_slots = 6, 6, 3
    max_health = torch.tensor([0, 35, 28, 25, 27, 32], dtype=torch.float)
    damage = torch.zeros(num_types, num_abilities, dtype=torch.float)
    turn_cd = torch.zeros(num_types, num_abilities, dtype=torch.long)
    target_mask = torch.zeros(num_types, num_abilities, num_slots, dtype=torch.bool)
    effect_type = torch.zeros(num_types, num_abilities, dtype=torch.long)

    # Guerrero tipo 1, habilidad 0: ataque a los 3 slots, daño 10
    damage[1, 0] = 10.0
    target_mask[1, 0] = torch.tensor([True, True, True])
    effect_type[1, 0] = int(EffectType.ATTACK)

    # Guerrero tipo 1, habilidad 1: defensa total
    effect_type[1, 1] = int(EffectType.DEFEND_FULL)

    # Guerrero tipo 1, habilidad 2: defensa parcial
    effect_type[1, 2] = int(EffectType.DEFEND_HALF)

    # Guerrero tipo 2, habilidad 0: ataque solo a slot 1, daño 7
    damage[2, 0] = 7.0
    target_mask[2, 0] = torch.tensor([False, True, False])
    effect_type[2, 0] = int(EffectType.ATTACK)

    return resolveAction(max_health, damage, turn_cd, target_mask, effect_type)


def _reference_resolve_action_attack(resolver, actors, ability_pool_idx, enemy_disposition, enemy_health, enemy_alive, enemy_actions, enemy_instance_abilities):
    """Copia exacta del algoritmo ORIGINAL con bucle, para comparar."""
    would_be_damage = resolver.damage_por_tipo_habilidad[actors, ability_pool_idx]
    target_mask = resolver.target_mask_por_tipo_habilidad[actors, ability_pool_idx]

    enemy_ability_pool_idx = enemy_instance_abilities.gather(2, enemy_actions.clamp(0, 3).unsqueeze(-1)).squeeze(-1)
    enemy_effect_type = resolver.effect_type_por_tipo_habilidad[enemy_disposition, enemy_ability_pool_idx]
    enemy_es_habilidad = (enemy_actions >= 0) & (enemy_actions <= 3)

    enemy_new_health = enemy_health.clone()
    damage_total = torch.zeros_like(would_be_damage)
    avoided_total = torch.zeros_like(would_be_damage)
    blocks_total = torch.zeros_like(would_be_damage)
    overkill_damage = torch.zeros_like(would_be_damage)
    kills_this_action = torch.zeros_like(would_be_damage)

    for slot in range(3):
        es_target = target_mask[:, slot] & enemy_alive[:, slot]
        full_block = (enemy_effect_type[:, slot] == EffectType.DEFEND_FULL) & enemy_es_habilidad[:, slot]
        half_block = (enemy_effect_type[:, slot] == EffectType.DEFEND_HALF) & enemy_es_habilidad[:, slot]

        hit_damage = torch.where(full_block, torch.zeros_like(would_be_damage),
                                  torch.where(half_block, would_be_damage / 2, would_be_damage))
        avoided = torch.where(full_block, would_be_damage,
                               torch.where(half_block, would_be_damage / 2, torch.zeros_like(would_be_damage)))
        blocked_flag = (full_block | half_block).float()

        hit_damage = torch.where(es_target, hit_damage, torch.zeros_like(hit_damage))
        avoided = torch.where(es_target, avoided, torch.zeros_like(avoided))
        blocked_flag = torch.where(es_target, blocked_flag, torch.zeros_like(blocked_flag))

        health_slot_before = enemy_new_health[:, slot]
        overkill_this_slot = torch.where(
            es_target & (health_slot_before > 0) & (hit_damage >= health_slot_before),
            hit_damage - health_slot_before, torch.zeros_like(hit_damage))
        overkill_damage += overkill_this_slot

        kill_this_slot = torch.where(
            es_target & (health_slot_before > 0) & (hit_damage >= health_slot_before),
            torch.ones_like(hit_damage), torch.zeros_like(hit_damage))
        kills_this_action += kill_this_slot

        enemy_new_health[:, slot] = torch.where(es_target, health_slot_before - hit_damage, health_slot_before)
        damage_total += hit_damage
        avoided_total += avoided
        blocks_total += blocked_flag

    enemy_new_alive = enemy_alive & (enemy_new_health > 0)
    return damage_total, blocks_total, enemy_new_health, enemy_new_alive, overkill_damage, kills_this_action


def _reference_check_if_targeted(resolver, pos, enemy_disposition, enemy_actions, enemy_instance_abilities, enemy_alive):
    N = enemy_disposition.shape[0]
    was_targeted = torch.zeros(N, dtype=torch.bool)
    pos = pos.view(-1).to(torch.long)
    row_idx = torch.arange(N)

    for e_slot in range(3):
        alive = enemy_alive[:, e_slot]
        action = enemy_actions[:, e_slot]
        is_attack_action = (action >= 0) & (action <= 3)
        is_valid = alive & is_attack_action
        action_clamped = action.clamp(min=0, max=3).view(-1)
        abilities = enemy_instance_abilities[:, e_slot]
        ability_idx = abilities.gather(1, action_clamped.unsqueeze(1)).squeeze(1)
        enemy_type = enemy_disposition[:, e_slot].view(-1)
        target_mask = resolver.target_mask_por_tipo_habilidad[enemy_type, ability_idx]
        target_mask_for_pos = target_mask[row_idx, pos]
        was_targeted = was_targeted | (is_valid & target_mask_for_pos)

    return was_targeted


def _random_case(N, seed):
    torch.manual_seed(seed)
    actors = torch.randint(1, 3, (N,))
    ability_pool_idx = torch.randint(0, 3, (N,))
    enemy_disposition = torch.randint(0, 3, (N, 3))
    enemy_health = torch.rand(N, 3) * 30
    enemy_alive = torch.rand(N, 3) > 0.2
    enemy_actions = torch.randint(-1, 4, (N, 3))
    enemy_instance_abilities = torch.randint(0, 3, (N, 3, 4))
    return actors, ability_pool_idx, enemy_disposition, enemy_health, enemy_alive, enemy_actions, enemy_instance_abilities


def test_resolve_action_attack_matches_reference_multiple_seeds():
    resolver = _make_resolver()
    for seed in range(10):
        for N in (1, 5, 50):
            case = _random_case(N, seed)
            out_new = resolver._resolve_action_attack(*case)
            out_ref = _reference_resolve_action_attack(resolver, *case)
            for a, b in zip(out_new, out_ref):
                assert torch.allclose(a, b, atol=1e-5), f"Mismatch seed={seed} N={N}"


def test_check_if_targeted_matches_reference_multiple_seeds():
    resolver = _make_resolver()
    for seed in range(10):
        for N in (1, 5, 50):
            actors, ability_pool_idx, enemy_disposition, enemy_health, enemy_alive, enemy_actions, enemy_instance_abilities = _random_case(N, seed)
            pos = torch.randint(0, 3, (N,))
            out_new = resolver._check_if_targeted(pos, enemy_disposition, enemy_actions, enemy_instance_abilities, enemy_alive)
            out_ref = _reference_check_if_targeted(resolver, pos, enemy_disposition, enemy_actions, enemy_instance_abilities, enemy_alive)
            assert torch.equal(out_new, out_ref), f"Mismatch seed={seed} N={N}"


def test_check_if_targeted_edge_case_n1():
    """Caso límite explícito N=1, el que motivaba las ramas defensivas eliminadas."""
    resolver = _make_resolver()
    pos = torch.tensor([1])
    enemy_disposition = torch.tensor([[1, 2, 0]])
    enemy_actions = torch.tensor([[0, 0, -1]])
    enemy_instance_abilities = torch.zeros(1, 3, 4, dtype=torch.long)
    enemy_alive = torch.tensor([[True, True, False]])

    result = resolver._check_if_targeted(pos, enemy_disposition, enemy_actions, enemy_instance_abilities, enemy_alive)
    assert result.shape == (1,)
    # tipo 1 hab 0 ataca [0,1,2] -> True; tipo 2 hab 0 ataca solo [1] -> también True
    assert result.item() == True