"""
Fábrica de datos de guerreros.

Proporciona la configuración estática de todos los tipos de guerreros disponibles.
En el futuro, esta configuración podría cargarse desde un archivo JSON/YAML.
"""
from typing import Dict

from AI.Environment.abilityData import AbilityData
from AI.Environment.warriorData import WarriorData


def get_warriors_classes() -> Dict[int, WarriorData]:
    """
    Construye y devuelve el diccionario de todos los tipos de guerreros.

    Returns:
        Diccionario {id: WarriorData} con los datos de cada guerrero.
    """
    # ------------------------------------------------------------
    # 1. KNIGHT (Tanque / Daño cuerpo a cuerpo)
    # ------------------------------------------------------------
    knight = WarriorData(
        id=1,
        name="Knight",
        max_health=35,
        speed=10,
        ability_pool=[
            AbilityData("Smite", 1, 10, [0], 1),      # Golpe fuerte al frente
            AbilityData("Guard Up", 2, 0, [], 1),    # Bloquea todo daño este turno
            AbilityData("Slice", 3, 7, [0, 1, 2], 0), # Ataque en área (repetible)
            AbilityData("Throw", 4, 9, [2], 0),       # Ataque a distancia (repetible)
        ]
    )

    # ------------------------------------------------------------
    # 2. ARCHER (Daño a distancia / Velocidad)
    # ------------------------------------------------------------
    archer = WarriorData(
        id=2,
        name="Archer",
        max_health=27,
        speed=18,
        ability_pool=[
            AbilityData("Arrow", 1, 10, [1], 0),       # Ataque al centro
            AbilityData("Heal", 2, 10, [], 1),        # Curación personal (6 de vida)
            AbilityData("Arrow2", 3, 9, [2], 0),      # Ataque a la retaguardia
            AbilityData("Rain", 4, 8, [0, 1, 2], 1), # Lluvia de flechas (todas las posiciones)
        ]
    )

    # ------------------------------------------------------------
    # 3. ROGUE (Alto daño / Baja vida / Muy rápido)
    # ------------------------------------------------------------
    rogue = WarriorData(
        id=3,
        name="Rogue",
        max_health=23,
        speed=20,
        ability_pool=[
            AbilityData("BackAttack", 1, 13, [2], 1), # Ataque crítico a la retaguardia
            AbilityData("Hide", 2, 0, [], 1),        # Se oculta (bloquea daño)
            AbilityData("PoisonGas", 3, 9, [0, 1, 2], 1), # Veneno en área
            AbilityData("Knife", 4, 8, [0], 0),       # Ataque repetible al frente
        ]
    )

    # ------------------------------------------------------------
    # 4. WIZARD (Daño mágico / Versátil)
    # ------------------------------------------------------------
    wizard = WarriorData(
        id=4,
        name="Wizard",
        max_health=25,
        speed=12,
        ability_pool=[
            AbilityData("Magic Missile", 1, 6, [0, 1, 2], 0), # Daño bajo pero seguro
            AbilityData("Zap", 2, 12, [1, 2], 1),       # Rayo a centro/retaguardia
            AbilityData("FireBall", 3, 9, [0, 1, 2], 1), # Bola de fuego en área
            AbilityData("StaffAttack", 4, 8, [0], 0),   # Ataque físico repetible
        ]
    )

    # ------------------------------------------------------------
    # 5. CLERIC (Curación / Apoyo)
    # ------------------------------------------------------------
    cleric = WarriorData(
        id=5,
        name="Cleric",
        max_health=30,
        speed=14,
        ability_pool=[
            AbilityData("Charge", 1, 8, [0, 1], 0),    # Ataque modesto al frente/centro
            AbilityData("HealAll", 2, 11, [], 1),     # Cura a todos los aliados
            AbilityData("Defend", 3, 0, [], 1),      # Reduce daño recibido 50% este turno
            AbilityData("Light", 4, 7, [0, 1, 2], 1), # Ataque de luz en área
        ]
    )

    # ------------------------------------------------------------
    # Construcción del diccionario y validación
    # ------------------------------------------------------------
    warrior_classes = {
        knight.id: knight,
        archer.id: archer,
        rogue.id: rogue,
        wizard.id: wizard,
        cleric.id: cleric,
    }

    # Validación de que todos los IDs estén en el rango esperado
    if len(warrior_classes) != 5:
        raise RuntimeError("No se cargaron todos los guerreros correctamente.")

    return warrior_classes


# ================================================================
# FUTURO: Carga desde JSON/YAML (cuando se añadan más guerreros)
# ================================================================
# import json
# def get_warriors_from_json(path: str) -> Dict[int, WarriorData]:
#     with open(path, "r") as f:
#         data = json.load(f)
#     warriors = {}
#     for k, v in data.items():
#         abilities = [AbilityData(**a) for a in v.pop("abilities")]
#         warriors[int(k)] = WarriorData(**v, abilities=abilities)
#     return warriors