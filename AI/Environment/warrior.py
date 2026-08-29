"""
Instancia de un guerrero en una partida (modo no vectorizado).

Esta clase se usa principalmente para el jugador humano (PlayerNoAIV)
y para pruebas unitarias. El entorno vectorizado trabaja directamente con tensores.
"""
from typing import List, Optional

from AI.Environment.abilityData import AbilityData
from AI.Environment.warriorData import WarriorData


class Warrior:
    """
    Representa una instancia viva de un guerrero en el campo de batalla.

    Gestiona la salud actual y los cooldowns de las habilidades (visibles para el humano).
    """

    def __init__(self, warrior_data: WarriorData,abilities) -> None:
        """
        Args:
            warrior_data: Los datos estáticos del tipo de guerrero.
        """
        self.warrior_data: WarriorData = warrior_data
        self.health: int = warrior_data.max_health
        # cooldown_abilities[i] = True si la habilidad i está en enfriamiento
        self.cooldown_abilities: List[int] = [0] * 4
        #Lista de ids de abilidad del 0 al maximo de habilidades por guerrero
        self.abilities: List[int] = abilities 