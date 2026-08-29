"""
Definición de los datos de una habilidad individual.

Las habilidades son la base de las acciones de los guerreros durante el turno.
"""
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class AbilityData:
    """
    Datos inmutables de una habilidad.

    Attributes:
        name: Nombre de la habilidad.
        id: Identificador numérico (1-4, dentro del conjunto de habilidades del guerrero).
        damage: Cantidad de daño o curación que inflige (0 para habilidades defensivas).
        target_positions: Lista de posiciones (0, 1, 2) que puede atacar/curar. Vacío si es auto-target.
        can_repeat: Si se puede usar en el mismo turno (False = entra en cooldown por 1 turno).
    """
    name: str
    id: int
    damage: int
    target_positions: List[int]
    turn_cd: float
    effect_type: EffectType
    
    
from enum import Enum, auto

class EffectType(Enum):
    ATTACK = auto()
    SELF_HEAL = auto()
    TEAM_HEAL = auto()
    DEFEND = auto()