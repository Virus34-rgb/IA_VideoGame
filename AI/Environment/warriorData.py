"""
Definición de los datos estáticos de un tipo de guerrero.
"""
from dataclasses import dataclass, field
from typing import List

from AI.Environment.abilityData import AbilityData
import constants


@dataclass(frozen=True)
class WarriorData:
    """
    Datos inmutables de un tipo de guerrero (Knight, Archer, etc.).

    Attributes:
        id: Identificador único del tipo de guerrero (1..WARRIOR_QUANTITY).
        name: Nombre del guerrero.
        max_health: Vida máxima base.
        speed: Velocidad (determina el orden de turno).
        ability_pool: Catálogo de habilidades del que se sortean las 4 equipadas por instancia.
    """
    id: int
    name: str
    max_health: int
    speed: int
    ability_pool: List[AbilityData] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Valida que la pool tenga al menos las habilidades necesarias para equipar."""
        if len(self.ability_pool) < constants.ABILITIES_PER_WARRIOR:
            raise ValueError(
                f"Warrior {self.name} tiene una pool de {len(self.ability_pool)} "
                f"habilidades, se necesitan al menos {constants.ABILITIES_PER_WARRIOR}."
            )