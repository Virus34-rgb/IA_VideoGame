"""
Test de regresión dorado contra PlayerRusherV(aggression=0.5).

Ejecuta un lote determinista (seed fija, N pequeño, redes en eval, sin
exploración residual) contra el rusher con agresión fija 0.5, y compara el
stats2.txt resultante contra un archivo dorado commiteado en
`tests/golden/stats_rusher_aggr_05_seed42.txt`.

Qué detecta:
  - Cambios en fórmulas de daño/curas/bloqueos
  - Cambios en el orden de turno (speeds)
  - Cambios en action masking
  - Cambios en encoding de observación
  - Cambios en el cálculo de reward
  - Cambios en la política del rusher

Qué NO detecta (a propósito):
  - Mejoras/regresiones de entrenamiento (el test no entrena, solo evalúa
    una red inicializada aleatoriamente con seed fija).
  - Cambios de hiperparámetros que no alteran la inferencia.

Qué hacer cuando falla:
  1. Leer el diff que imprime pytest. Cada línea es una métrica con el
     valor golden, el valor actual y el delta.
  2. Si el cambio es INTENCIONAL (ej. has tocado resolve_actions), corre
     `python tests/regenerate_golden.py` y commitea el nuevo golden JUNTO
     con el cambio de código.
  3. Si NO entiendes por qué cambió, es un bug. Investiga.

Determinismo:
  El test es sensible a CUALQUIER cambio que consuma RNG de forma distinta.
  Eso es intencional — sirve como filtro estricto de "algo cambió".
  Los bloques que consumen RNG (init de redes, init de castillos, política
  del rusher, política del agente) están aislados con reseeds explícitos
  para que un cambio en uno no contamine a los otros.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from AI.Agent.opponent_poolV import NullOpponentPool


# ── Configuración del test ──
# NO TOCAR sin regenerar el golden.
TEST_SEED = 42
TEST_N = 64                    # partidas paralelas
TEST_BATCHES = 1               # lotes
TEST_RUSHER_AGGRESSION = 0.5
TEST_MAX_TURNS = 10            # partidas cortas → test rápido

GOLDEN_PATH = Path(__file__).parent / "golden" / "stats_rusher_aggr_05_seed42.txt"

# ── Tolerancias por métrica ──
# Contadores enteros: exactos (tol=0).
# Ratios/medias: absorbemos redondeo de float32 con tolerancias pequeñas.
DEFAULT_TOLERANCE = 1e-4
TOLERANCES: dict[str, float] = {
    # RESULTADOS
    "Partidas": 0,
    "Victorias P1": 0,
    "Victorias P2": 0,
    "Empates": 0,
    "Terminadas por muerte": 0,
    "Terminadas por límite": 0,
    "Turnos totales": 0,
    "Win ratio P1 (sin empates)": 1e-2,
    "Win ratio P2 (sin empates)": 1e-2,
    "Turnos medios por partida": 1e-2,
    # DAÑO / BAJAS / HEAL
    "Daño medio P1": 1e-2,
    "Daño medio P2": 1e-2,
    "Bajas medias P1": 1e-2,
    "Bajas medias P2": 1e-2,
    "Heal medio P1": 1e-2,
    "Heal medio P2": 1e-2,
    "Reward media P1": 1e-2,
    "Reward media P2": 1e-2,
    # RUSHER (dentro de RESULTADOS VS RUSHER)
    "PARTIDAS VS RUSHER": 0,
    "Victorias IA": 0,
    "Victorias Rusher": 0,
}


def _parse_stats(path: Path) -> dict[str, float]:
    """
    Parsea un stats*.txt a {nombre_metrica: primer_valor_numerico}.

    Ignora líneas sin ':' o sin números, y cabeceras decorativas (líneas
    que empiezan por '=' o '-'). Toma SOLO el primer número de cada línea
    (los '(2nd)' como porcentajes quedan fuera — si los necesitas, usa el
    parser de producción `stats_parsing.parse_stats_file`).
    """
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        name, rest = line.split(":", 1)
        name = name.strip()
        if not name or name.startswith("=") or name.startswith("-"):
            continue
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", rest)
        if not nums:
            continue
        try:
            out[name] = float(nums[0])
        except ValueError:
            continue
    return out


def _run_deterministic_eval(tmp_path: Path) -> Path:
    """
    Ejecuta el eval determinista contra el rusher 0.5 y devuelve la ruta del
    stats2.txt generado.

    Todos los parámetros que afectan al comportamiento se fijan aquí. Cada
    bloque que consume RNG (init de redes, init de castillos) arranca con
    el MISMO seed, para que un cambio en un bloque no contamine a los otros.
    """
    import constants

    # ── Parámetros deterministas ──
    constants.SEED = TEST_SEED
    constants.N_BATCH = TEST_N
    constants.MAX_TURNS = TEST_MAX_TURNS
    constants.EPSILON_RESIDUAL = 0.0
    constants.EPSILON_SEL_MIN = 0.0
    constants.USE_WANDB = False
    constants.USE_TORCH_COMPILE = False
    constants.PROFILE_CPROFILE = False
    constants.PROFILE_TORCH = False
    constants.RESET_IN_DECISIONS = False   # no resamplear ruido en cada decisión
    constants.USE_PROFILE_CONDITIONED_REWARD = False  # reward con pesos constantes

    # Un solo hilo: elimina cualquier orden de reducción no determinista
    torch.set_num_threads(1)
    torch.manual_seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    random.seed(TEST_SEED)

    from AI.Environment.vectorizedEnvironment import VectorizedEnvironment
    from AI.Agent.playerAIV import PlayerAIV
    from AI.Agent.player_rusher import PlayerRusherV
    from AI.Agent.trainerV import TrainerV

    # ── Construcción aislada por bloques ──
    # Nota: reseedeamos antes de cada constructor que consume RNG, para que
    # los tres bloques (entorno, redes, castillos) sean independientes entre
    # sí. Si añades un bloque nuevo con RNG, añade su propio reseed.

    torch.manual_seed(TEST_SEED)
    env = VectorizedEnvironment(TEST_N)                # sin RNG relevante

    torch.manual_seed(TEST_SEED)
    player1 = PlayerAIV(TEST_N, env, use_replay=False)  # init de redes

    torch.manual_seed(TEST_SEED)
    rusher = PlayerRusherV(TEST_N, env)                # tablas estáticas

    torch.manual_seed(TEST_SEED)
    stats2_path = tmp_path / "stats2.txt"
    # TrainerV construye los castillos → sample_abilities_batch consume RNG.
    trainer = TrainerV(
        player1, rusher, rusher,
        env, opponent_pool=NullOpponentPool(),
        train_batches=0, eval_batches=TEST_BATCHES,
        pathp1_1=str(tmp_path / "none_p1_sel.pth"),
        pathp1_2=str(tmp_path / "none_p1_turn.pth"),
        pathp2_1=str(tmp_path / "none_p2_sel.pth"),
        pathp2_2=str(tmp_path / "none_p2_turn.pth"),
        path_stats=str(tmp_path / "stats.txt"),
        path_stats2=str(stats2_path),
        logger=None,
        snapshot_every=0,
        progress_every=0,
    )

    # ── Redes en eval: NoisyLinear usa weight_mu (sin ruido) ──
    player1.selection_network.eval()
    player1.turn_network.eval()
    if hasattr(player1, "target_selection_network"):
        player1.target_selection_network.eval()
    if hasattr(player1, "target_turn_network"):
        player1.target_turn_network.eval()

    # ── Ejecutar eval vs rusher 0.5 ──
    trainer.evaluate(fixed_rusher_aggression=TEST_RUSHER_AGGRESSION)

    return stats2_path


# Si no hay golden, el test se marca como skipped con un mensaje útil.
# Generar uno con: python tests/regenerate_golden.py
@pytest.mark.skipif(
    not GOLDEN_PATH.exists(),
    reason=(
        f"Golden no existe en {GOLDEN_PATH}. "
        "Genera uno con: python tests/regenerate_golden.py"
    ),
)
def test_regression_rusher_05(tmp_path):
    """
    Regresión dorada: 1 lote determinista vs rusher agresión 0.5.

    Este test entrena CERO. Solo evalúa una red inicializada con seed fija
    contra un oponente determinista. Si el golden falla, algo cambió en el
    motor — no en el entrenamiento.
    """
    stats_path = _run_deterministic_eval(tmp_path)

    current = _parse_stats(stats_path)
    golden = _parse_stats(GOLDEN_PATH)

    assert current, f"No se generó stats en {stats_path} (¿falló el run?)"
    assert golden, f"El golden {GOLDEN_PATH} está vacío o ilegible."

    common_keys = sorted(set(current) & set(golden))
    assert common_keys, (
        "No hay métricas en común entre golden y run actual. "
        "¿Cambió el formato de stats2.txt?"
    )

    # ── Comparación con tolerancia ──
    diffs: list[str] = []
    for key in common_keys:
        tol = TOLERANCES.get(key, DEFAULT_TOLERANCE)
        got = current[key]
        exp = golden[key]
        delta = got - exp
        if abs(delta) > tol:
            diffs.append(
                f"  {key}: golden={exp!r}, actual={got!r}, "
                f"delta={delta:+.6f} (tol={tol})"
            )

    # Métricas que existían en el golden y desaparecieron → regresión de formato
    missing = sorted(set(golden) - set(current))
    if missing:
        diffs.append(f"  Métricas del golden ausentes en el run: {missing}")

    if diffs:
        msg = (
            "Regresión detectada contra el golden del rusher 0.5.\n"
            f"Golden: {GOLDEN_PATH}\n"
            "Diferencias:\n" + "\n".join(diffs) + "\n\n"
            "Si el cambio es INTENCIONAL, regenera el golden con:\n"
            "    python tests/regenerate_golden.py\n"
            "y commitea el nuevo archivo JUNTO con el cambio de código.\n\n"
            "Si NO entiendes por qué cambió, es un bug. Investiga antes de "
            "regenerar."
        )
        pytest.fail(msg)