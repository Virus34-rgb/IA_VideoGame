


import torch

import constants


def decidir_compra_batch(castle,warrior_most_use):
    hay_hueco = (~castle.castle_alive).any(dim=1)
    puede_pagar = castle.gold >= constants.COST_COMPRA
    mask_compra = hay_hueco & puede_pagar
    first_missing = find_first_missing_type(castle.castle_types, castle.castle_alive, constants.WARRIOR_QUANTITY)
    tipo_elegido = torch.where(first_missing > 0,first_missing,warrior_most_use)
    return mask_compra,tipo_elegido
    
    
def find_first_missing_type(castle_types: torch.Tensor, castle_alive: torch.Tensor, num_types: int) -> torch.Tensor:
    """
    Para cada fila de castle_types, devuelve el primer tipo (1..num_types) SIN NINGUNA
    instancia VIVA. Si todos los tipos tienen al menos una instancia viva, devuelve 0.
    Args:
        castle_types: (N, MAX_CASTLE_SIZE) con valores 0..num_types
        castle_alive: (N, MAX_CASTLE_SIZE) bool, True si esa instancia sigue viva
        num_types: número de tipos de guerreros a garantizar vivos (ej. 5)
    Returns:
        first_missing: (N,) entero, el primer tipo sin instancias vivas, o 0 si todos
        los tipos tienen al menos una viva.
    """
    N = castle_types.shape[0]

    present = torch.zeros(N, num_types, dtype=torch.bool, device=castle_types.device)

    for t in range(num_types):
        warrior_id = t + 1
        present[:, t] = ((castle_types == warrior_id) & castle_alive).any(dim=1)

    missing_mask = ~present
    all_present = ~missing_mask.any(dim=1)

    first_missing_idx = torch.argmax(missing_mask.int(), dim=1)  # (N,)

    first_missing_id = first_missing_idx + 1

    first_missing_id = torch.where(all_present, torch.zeros_like(first_missing_id), first_missing_id)

    return first_missing_id