"""
Definición de los datos estáticos de un tipo de guerrero.

Contiene las estadísticas base y la lista de habilidades que puede poseer.
"""
from dataclasses import dataclass, field
from typing import List

from AI.Environment.abilityData import AbilityData


@dataclass(frozen=True)
class WarriorData:
    """
    Datos inmutables de un tipo de guerrero (Knight, Archer, etc.).

    Attributes:
        id: Identificador único del tipo de guerrero (1..WARRIOR_QUANTITY).
        name: Nombre del guerrero.
        max_health: Vida máxima base.
        speed: Velocidad (determina el orden de turno).
        abilities: Lista de 4 habilidades que posee (de la pool de 7 en el futuro).
    """
    id: int
    name: str
    max_health: int
    speed: int
    ability_pool: List[AbilityData] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validación básica: asegura que el guerrero tenga exactamente 4 habilidades."""
        if len(self.ability_pool) <= 4:
            raise ValueError(f"Warrior {self.name} debe tener 4 habilidades, tiene {len(self.abilities)}")