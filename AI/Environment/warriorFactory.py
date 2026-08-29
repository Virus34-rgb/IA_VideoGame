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
        max_health=27,
        speed=10,
        abilities=[
            AbilityData("Smite", 1, 15, [0], False),      # Golpe fuerte al frente
            AbilityData("Guard Up", 2, 0, [], False),    # Bloquea todo daño este turno
            AbilityData("Slice", 3, 5, [0, 1, 2], True), # Ataque en área (repetible)
            AbilityData("Throw", 4, 12, [2], True),       # Ataque a distancia (repetible)
        ]
    )

    # ------------------------------------------------------------
    # 2. ARCHER (Daño a distancia / Velocidad)
    # ------------------------------------------------------------
    archer = WarriorData(
        id=2,
        name="Archer",
        max_health=19,
        speed=18,
        abilities=[
            AbilityData("Arrow", 1, 11, [1], True),       # Ataque al centro
            AbilityData("Heal", 2, 8, [], False),        # Curación personal (6 de vida)
            AbilityData("Arrow2", 3, 9, [2], True),      # Ataque a la retaguardia
            AbilityData("Rain", 4, 14, [0, 1, 2], False), # Lluvia de flechas (todas las posiciones)
        ]
    )

    # ------------------------------------------------------------
    # 3. ROGUE (Alto daño / Baja vida / Muy rápido)
    # ------------------------------------------------------------
    rogue = WarriorData(
        id=3,
        name="Rogue",
        max_health=15,
        speed=20,
        abilities=[
            AbilityData("BackAttack", 1, 18, [2], False), # Ataque crítico a la retaguardia
            AbilityData("Hide", 2, 0, [], False),        # Se oculta (bloquea daño)
            AbilityData("PoisonGas", 3, 13, [0, 1, 2], False), # Veneno en área
            AbilityData("Knife", 4, 10, [0], True),       # Ataque repetible al frente
        ]
    )

    # ------------------------------------------------------------
    # 4. WIZARD (Daño mágico / Versátil)
    # ------------------------------------------------------------
    wizard = WarriorData(
        id=4,
        name="Wizard",
        max_health=17,
        speed=12,
        abilities=[
            AbilityData("Magic Missile", 1, 7, [0, 1, 2], True), # Daño bajo pero seguro
            AbilityData("Zap", 2, 14, [1, 2], False),       # Rayo a centro/retaguardia
            AbilityData("FireBall", 3, 11, [0, 1, 2], False), # Bola de fuego en área
            AbilityData("StaffAttack", 4, 8, [0], True),   # Ataque físico repetible
        ]
    )

    # ------------------------------------------------------------
    # 5. CLERIC (Curación / Apoyo)
    # ------------------------------------------------------------
    cleric = WarriorData(
        id=5,
        name="Cleric",
        max_health=21,
        speed=14,
        abilities=[
            AbilityData("Charge", 1, 10, [0, 1], True),    # Ataque modesto al frente/centro
            AbilityData("HealAll", 2, 9, [], False),     # Cura a todos los aliados
            AbilityData("Defend", 3, 0, [], False),      # Reduce daño recibido 50% este turno
            AbilityData("Light", 4, 9, [0, 1, 2], False), # Ataque de luz en área
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