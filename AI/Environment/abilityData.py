"""
Definición de los datos de una habilidad individual.
"""
from dataclasses import dataclass
from typing import List
from enum import IntEnum


class EffectType(IntEnum):
    """
    Tipo de efecto de una habilidad. Sustituye a la antigua inferencia implícita
    por (tipo de guerrero, índice de botón) — dejó de ser válida en cuanto las 4
    habilidades equipadas pasaron a sortearse de una pool más grande.
    """
    ATTACK = 0
    SELF_HEAL = 1
    TEAM_HEAL = 2
    DEFEND_FULL = 3 
    DEFEND_HALF = 4


@dataclass(frozen=True)
class AbilityData:
    """
    Datos inmutables de una habilidad.

    Attributes:
        name: Nombre de la habilidad.
        id: Identificador local dentro de la pool del guerrero (0..pool_size-1).
        damage: Cantidad de daño o curación que inflige (0 para defensivas).
        target_positions: Posiciones (0,1,2) que puede atacar/curar. Vacío si autotarget.
        turn_cd: Turnos de cooldown tras usarla (0 = repetible cada turno).
        effect_type: Tipo de efecto (ataque, autocura, cura de equipo, defensa).
    """
    name: str
    id: int
    damage: int
    target_positions: List[int]
    turn_cd: int
    effect_type: EffectType