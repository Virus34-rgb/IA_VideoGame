"""
Test de equivalencia funcional para el swap de movimiento vectorizado.
Compara la implementación NUEVA (_swap_by_position) contra una implementación
de REFERENCIA escrita con bucle Python explícito (lenta pero obviamente correcta),
sobre distintos casos: mover derecha, izquierda, en los bordes, sin movimiento.
"""
import torch
from AI.Environment.resolve_actions import resolveAction


def _reference_swap(tensor, pos, pos_destino, mask):
    """Implementación de referencia con bucle Python fila a fila."""
    result = tensor.clone()
    N = tensor.shape[0]
    for i in range(N):
        if mask[i]:
            a, b = pos[i].item(), pos_destino[i].item()
            result[i, a], result[i, b] = tensor[i, b].clone(), tensor[i, a].clone()
    return result


def _make_resolver():
    max_health = torch.zeros(6, dtype=torch.float)
    damage = torch.zeros(6, 6, dtype=torch.float)
    turn_cd = torch.zeros(6, 6, dtype=torch.long)
    target_mask = torch.zeros(6, 6, 3, dtype=torch.bool)
    effect_type = torch.zeros(6, 6, dtype=torch.long)
    return resolveAction(max_health, damage, turn_cd, target_mask, effect_type)


def test_swap_2d_tensor_matches_reference():
    resolver = _make_resolver()
    N = 20
    torch.manual_seed(0)
    tensor = torch.randint(0, 5, (N, 3))
    pos = torch.randint(0, 3, (N,))
    pos_destino = torch.randint(0, 3, (N,))
    mask = torch.rand(N) > 0.5

    out_new = resolver._swap_by_position(tensor, pos, pos_destino, mask)
    out_ref = _reference_swap(tensor, pos, pos_destino, mask)

    assert torch.equal(out_new, out_ref)


def test_swap_3d_tensor_matches_reference():
    """Cooldowns/habilidades tienen forma (N,3,4)."""
    resolver = _make_resolver()
    N = 15
    torch.manual_seed(1)
    tensor = torch.randint(0, 3, (N, 3, 4))
    pos = torch.randint(0, 3, (N,))
    pos_destino = torch.randint(0, 3, (N,))
    mask = torch.rand(N) > 0.5

    def _reference_swap_3d(t, p, pd, m):
        result = t.clone()
        for i in range(N):
            if m[i]:
                a, b = p[i].item(), pd[i].item()
                result[i, a, :], result[i, b, :] = t[i, b, :].clone(), t[i, a, :].clone()
        return result

    out_new = resolver._swap_by_position(tensor, pos, pos_destino, mask)
    out_ref = _reference_swap_3d(tensor, pos, pos_destino, mask)

    assert torch.equal(out_new, out_ref)


def test_no_move_is_noop():
    """mask=False en toda la fila -> tensor sin cambios."""
    resolver = _make_resolver()
    tensor = torch.randn(5, 3)
    pos = torch.tensor([0, 1, 2, 0, 1])
    pos_destino = torch.tensor([1, 2, 0, 1, 2])
    mask = torch.zeros(5, dtype=torch.bool)

    out = resolver._swap_by_position(tensor, pos, pos_destino, mask)
    assert torch.equal(out, tensor)


def test_same_position_is_noop_even_if_masked():
    """pos_destino == pos (caso 'sin movimiento' codificado como destino=pos)
    debe ser no-op aunque mask sea True."""
    resolver = _make_resolver()
    tensor = torch.tensor([[10., 20., 30.]])
    pos = torch.tensor([1])
    pos_destino = torch.tensor([1])
    mask = torch.tensor([True])

    out = resolver._swap_by_position(tensor, pos, pos_destino, mask)
    assert torch.equal(out, tensor)


def test_full_resolve_action_movement_boundaries():
    """Test de integración: movimiento en los bordes (pos=0 no puede ir a
    la izquierda, pos=2 no puede ir a la derecha) usando resolve_action_movement
    completo, comparando disposición/salud antes y después."""
    resolver = _make_resolver()
    N = 4
    actors = torch.tensor([1, 1, 1, 1])
    own_disposition = torch.tensor([[1, 2, 3]] * N)
    own_health = torch.tensor([[10., 20., 30.]] * N)
    own_cooldowns = torch.zeros(N, 3, 4, dtype=torch.long)
    own_instance_abilities = torch.zeros(N, 3, 4, dtype=torch.long)
    own_castle_slots = torch.zeros(N, 3, dtype=torch.long)
    own_alive = torch.ones(N, 3, dtype=torch.bool)
    enemy_disposition = torch.zeros(N, 3, dtype=torch.long)
    enemy_alive = torch.zeros(N, 3, dtype=torch.bool)
    enemy_actions = torch.full((N, 3), -1)
    enemy_instance_abilities = torch.zeros(N, 3, 4, dtype=torch.long)

    # caso 0: pos=0 intenta moverse izquierda (acción 6) -> inválido, no debe mover
    # caso 1: pos=2 intenta moverse derecha (acción 5) -> inválido, no debe mover
    # caso 2: pos=0 se mueve derecha (acción 5) -> válido
    # caso 3: pos=1 se mueve izquierda (acción 6) -> válido
    actions_actor = torch.tensor([6, 5, 5, 6])
    pos = torch.tensor([0, 2, 0, 1])

    result = resolver._resolve_action_movement(
        actors, own_disposition, own_health, own_cooldowns,
        own_instance_abilities, own_castle_slots, own_alive, actions_actor, pos,
        enemy_disposition, enemy_alive, enemy_actions, enemy_instance_abilities,
    )
    moved, new_disp, new_health, *_ = result

    # casos 0 y 1 son inválidos por borde -> moved debe ser 0 y disposición sin cambio
    assert moved[0].item() == 0.0
    assert moved[1].item() == 0.0
    assert torch.equal(new_disp[0], own_disposition[0])
    assert torch.equal(new_disp[1], own_disposition[1])

    # caso 2: pos 0 <-> 1 se intercambian
    assert moved[2].item() == 1.0
    assert new_disp[2].tolist() == [2, 1, 3]
    assert new_health[2].tolist() == [20., 10., 30.]

    # caso 3: pos 1 <-> 0 se intercambian
    assert moved[3].item() == 1.0
    assert new_disp[3].tolist() == [2, 1, 3]