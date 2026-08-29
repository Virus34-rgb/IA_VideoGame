"""
Utilidades para el sorteo de habilidades equipadas por instancia de guerrero.
"""
import random
from typing import List

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