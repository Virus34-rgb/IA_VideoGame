"""
Utilidades para el sorteo de habilidades equipadas por instancia de guerrero.
"""
import random
from typing import List

import torch

from AI.Environment.warriorData import WarriorData
import constants


def sample_abilities(warrior_data: WarriorData, k: int = constants.ABILITIES_PER_WARRIOR) -> List[int]:
    """
    Sortea k índices locales (sin repetición) de la pool de habilidades de un
    tipo de guerrero. Los índices son locales a warrior_data.ability_pool.
    """
    pool_size = len(warrior_data.ability_pool)
    if pool_size < k:
        raise ValueError(
            f"{warrior_data.name} tiene una pool de {pool_size} habilidades, "
            f"se necesitan al menos {k}."
        )
    return random.sample(range(pool_size), k)

def sample_abilities_batch(pool_size: int, N: int, k: int = constants.ABILITIES_PER_WARRIOR) -> torch.Tensor:
    """Sortea, para cada una de las N partidas, k índices sin repetir de [0, pool_size)."""
    noise = torch.rand(N, pool_size)
    order = torch.argsort(noise, dim=1)
    return order[:, :k]   # (N, k)

def sample_abilities_batch_all_types(warriors_classes: dict, N: int, k: int = constants.ABILITIES_PER_WARRIOR) -> torch.Tensor:
    """Igual que sample_abilities_batch pero para todos los tipos a la vez -> (N, WARRIOR_QUANTITY, k)."""
    per_type = [
        sample_abilities_batch(len(warriors_classes[wid].ability_pool), N, k)
        for wid in sorted(warriors_classes.keys())
    ]
    return torch.stack(per_type, dim=1)