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

    def __init__(self, warrior_data: WarriorData) -> None:
        """
        Args:
            warrior_data: Los datos estáticos del tipo de guerrero.
        """
        self.warrior_data: WarriorData = warrior_data
        self.health: int = warrior_data.max_health
        # cooldown_abilities[i] = True si la habilidad i está en enfriamiento
        self.cooldown_abilities: List[bool] = [False] * 4

    def use_ability(self, pos: int) -> None:
        """
        Marca una habilidad como usada (activa su cooldown) si no es repetible.

        Args:
            pos: Índice de la habilidad usada (0-3).
        """
        ability = self.warrior_data.abilities[pos]
        if not ability.can_repeat:
            self.cooldown_abilities[pos] = True

    def usable_abilities(self) -> List[AbilityData]:
        """
        Devuelve la lista de habilidades que están disponibles (sin cooldown).

        Returns:
            Lista de objetos AbilityData disponibles para usar.
        """
        usable = []
        for idx, ability in enumerate(self.warrior_data.abilities):
            if not self.cooldown_abilities[idx]:
                usable.append(ability)
        return usable

    def reset_cooldowns(self) -> None:
        """Reinicia todos los cooldowns al final del turno (cooldowns de 1 turno)."""
        for i in range(len(self.cooldown_abilities)):
            if self.cooldown_abilities[i]:
                self.cooldown_abilities[i] = False

    def modify_health(self, damage: int, curation: int) -> None:
        """
        Modifica la salud del guerrero aplicando daño y curación.

        La salud se mantiene entre 0 y max_health.

        Args:
            damage: Daño recibido (positivo).
            curation: Curación recibida (positivo).
        """
        self.health = min(self.warrior_data.max_health, self.health - damage + curation)
        self.health = max(0, self.health)

    def reset_health(self) -> None:
        """Restaura la salud al máximo (para reinicios de partida)."""
        self.health = self.warrior_data.max_health