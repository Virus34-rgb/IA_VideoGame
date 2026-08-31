"""
Fábrica de datos de guerreros.
"""
from typing import Dict

from AI.Environment.abilityData import AbilityData, EffectType
from AI.Environment.warriorData import WarriorData


def get_warriors_classes() -> Dict[int, WarriorData]:
    knight = WarriorData(
        id=1, name="Knight", max_health=35, speed=10,
        ability_pool=[
            AbilityData("Smite", 0, 10, [1], 1, EffectType.ATTACK),
            AbilityData("Guard Up", 1, 0, [], 1, EffectType.DEFEND_FULL),
            AbilityData("Slice", 2, 7, [0, 1, 2], 0, EffectType.ATTACK),
            AbilityData("Throw", 3, 9, [2], 0, EffectType.ATTACK),
            AbilityData("Shield Bash", 4, 8, [0], 1, EffectType.ATTACK),
            AbilityData("Second Wind", 5, 10, [], 2, EffectType.SELF_HEAL),
        ]
    )

    archer = WarriorData(
        id=2, name="Archer", max_health=28, speed=18,
        ability_pool=[
            AbilityData("Arrow", 0, 10, [1], 0, EffectType.ATTACK),
            AbilityData("Heal", 1, 10, [], 1, EffectType.SELF_HEAL),
            AbilityData("Arrow2", 2, 9, [2], 0, EffectType.ATTACK),
            AbilityData("Rain", 3, 8, [0, 1, 2], 1, EffectType.ATTACK),
            AbilityData("Arrow Shield", 4, 0, [], 1, EffectType.DEFEND_FULL),
            AbilityData("Multi Shot", 5, 8, [0, 2], 1, EffectType.ATTACK),
        ]
    )

    rogue = WarriorData(
        id=3, name="Rogue", max_health=25, speed=20,
        ability_pool=[
            AbilityData("BackAttack", 0, 13, [2], 1, EffectType.ATTACK),
            AbilityData("Hide", 1, 0, [], 1, EffectType.DEFEND_FULL),
            AbilityData("PoisonGas", 2, 9, [0, 1, 2], 1, EffectType.ATTACK),
            AbilityData("Knife", 3, 8, [0], 0, EffectType.ATTACK),
            AbilityData("Ambush", 4, 16, [0], 2, EffectType.ATTACK),
            AbilityData("Vanish", 5, 0, [], 2, EffectType.DEFEND_FULL),
        ]
    )

    wizard = WarriorData(
        id=4, name="Wizard", max_health=27, speed=12,
        ability_pool=[
            AbilityData("Magic Missile", 0, 5, [0, 1, 2], 0, EffectType.ATTACK),
            AbilityData("Zap", 1, 10, [1, 2], 1, EffectType.ATTACK),
            AbilityData("FireBall", 2, 8, [0, 1, 2], 1, EffectType.ATTACK),
            AbilityData("StaffAttack", 3, 8, [0], 0, EffectType.ATTACK),
            AbilityData("Magic Shield", 4, 0, [], 1, EffectType.DEFEND_FULL),
            AbilityData("Arcane Surge", 5, 15, [1], 2, EffectType.ATTACK),
        ]
    )

    cleric = WarriorData(
        id=5, name="Cleric", max_health=32, speed=14,
        ability_pool=[
            AbilityData("Charge", 0, 8, [0, 1], 0, EffectType.ATTACK),
            AbilityData("HealAll", 1, 11, [], 2, EffectType.TEAM_HEAL),
            AbilityData("Defend", 2, 0, [], 1, EffectType.DEFEND_HALF),
            AbilityData("Light", 3, 7, [0, 1, 2], 1, EffectType.ATTACK),
            AbilityData("Holy Nova", 4, 9, [0, 1, 2], 1, EffectType.ATTACK),
            AbilityData("Renew", 5, 7, [], 1, EffectType.SELF_HEAL),
        ]
    )

    warrior_classes = {
        knight.id: knight, archer.id: archer, rogue.id: rogue,
        wizard.id: wizard, cleric.id: cleric,
    }
    if len(warrior_classes) != 5:
        raise RuntimeError("No se cargaron todos los guerreros correctamente.")
    return warrior_classes