"""
Instancia de un guerrero en una partida (modo no vectorizado, jugador humano).
"""
from typing import List

from AI.Environment.warriorData import WarriorData


class Warrior:
    def __init__(self, warrior_data: WarriorData, abilities: List[int]) -> None:
        """
        Args:
            warrior_data: Datos estáticos del tipo de guerrero.
            abilities: 4 índices locales (dentro de warrior_data.ability_pool) equipados.
        """
        self.warrior_data: WarriorData = warrior_data
        self.health: int = warrior_data.max_health
        self.cooldown_abilities: List[int] = [0] * 4
        self.abilities: List[int] = abilities   # CORREGIDO typo "abilites" -> "abilities"

    def use_ability(self, pos: int) -> None:
        pool_idx = self.abilities[pos]
        ability = self.warrior_data.ability_pool[pool_idx]   # CAMBIADO: antes self.warrior_data.abilities[pos] (no existe)
        if ability.turn_cd > 0:
            self.cooldown_abilities[pos] = ability.turn_cd

    def usable_abilities(self):
        usable = []
        for idx in range(4):
            if self.cooldown_abilities[idx] == 0:
                pool_idx = self.abilities[idx]
                usable.append(self.warrior_data.ability_pool[pool_idx])
        return usable

    def reset_cooldowns(self) -> None:
        for i in range(len(self.cooldown_abilities)):
            if self.cooldown_abilities[i] > 0:
                self.cooldown_abilities[i] = max(0, self.cooldown_abilities[i] - 1)

    def modify_health(self, damage: int, curation: int) -> None:
        self.health = min(self.warrior_data.max_health, self.health - damage + curation)
        self.health = max(0, self.health)

    def reset_health(self) -> None:
        self.health = self.warrior_data.max_health