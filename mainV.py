"""
Punto de entrada principal del proyecto Castle Game.

Este archivo se limita a: cargar config.yaml, aplicarlo sobre constants.py,
y decidir el modo de ejecución (comparación / multi-seed / single-run). Toda
la lógica de orquestación real vive en AI/Agent/orchestration/ (main_runner,
step_builder, run_spec, multi_seed_runner) y AI/Logging/ (parsing de stats,
comparación, log de experimentos) -- ver esos módulos para el detalle.
"""
import os

import torch

import constants
from AI.Agent.orchestration.main_runner import MainV
from AI.Agent.orchestration.multi_seed_runner import run_multi_seed
from AI.Agent.orchestration.run_spec import RunSpec, load_config_yaml, run_comparison, run_single
from AI.Agent.orchestration.seed_utils import set_seed
from AI.Agent.orchestration.step_builder import build_steps
from config import RunConfig

# Claves de config.yaml que NO son constantes globales (secciones
# estructuradas propias del modo RUN_COMPARISON) -- se excluyen del bucle
# setattr de _apply_yaml_overrides porque no tienen equivalente 1:1 en
# constants.py (son listas de dicts, no un valor escalar/lista simple).
NON_CONSTANT_YAML_KEYS = {"comparisons"}


def _apply_yaml_overrides(yaml_config: dict) -> None:
    """Aplica cada clave de config.yaml como constants.<CLAVE> = valor.
    Único punto del proyecto donde YAML muta constants.py en caliente -- ver
    AI.Logging.constants_diff.compute_diff() para verificar al final de un
    experimento que un override realmente se propagó (un `from constants
    import X` en otro módulo dejaría este setattr sin efecto sobre esa X)."""
    for key, value in yaml_config.items():
        if key in NON_CONSTANT_YAML_KEYS:
            continue
        if hasattr(constants, key):
            setattr(constants, key, value)
        else:
            print(f"Advertencia: {key} no existe en constants, se omite")


def _build_comparison_run_specs(yaml_config: dict) -> list:
    """Traduce la sección 'comparisons' de config.yaml a una lista de
    RunSpec, para el modo RUN_COMPARISON."""
    return [
        RunSpec(
            run_name=comp.get("run_name", "unnamed"),
            N=comp.get("N", constants.N_BATCH),
            train_batches=comp.get("train_batches", constants.TRAIN_EPISODES),
            eval_batches=comp.get("eval_batches", constants.EVAL_EPISODES),
            constants_overrides=comp.get("overrides", {}),
            seed=comp.get("seed", constants.SEED),
        )
        for comp in yaml_config["comparisons"]
    ]


if __name__ == "__main__":
    # Limitar threads BLAS/OMP: con N=2048 partidas vectorizadas, más threads
    # que núcleos físicos reales solo añade overhead de contención, no velocidad.
    torch.set_num_threads(2)
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"

    yaml_config = load_config_yaml()
    _apply_yaml_overrides(yaml_config)

    config = RunConfig(
        version=constants.VERSION,
        train_episodes=constants.TRAIN_EPISODES,
        eval_episodes=constants.EVAL_EPISODES,
        suffix=constants.RUN_NAME_SUFFIX,
    )

    # Los tres modos son mutuamente excluyentes -- prioridad: comparación >
    # multi-seed > single-run (misma prioridad que tenía el mainV.py original).
    if constants.RUN_COMPARISON and "comparisons" in yaml_config:
        run_comparison(config, _build_comparison_run_specs(yaml_config))

    elif constants.MULTI_SEED_ENABLED:
        run_multi_seed(config, constants.MULTI_SEED_LIST, baseline_dir=constants.MULTI_SEED_BASELINE)

    else:
        if constants.SEED is not None and constants.SEED != 'None':
            set_seed(constants.SEED)
        steps = build_steps(config)
        # N=1 obligatorio en modos con jugador humano (PlayerNoAIV solo
        # soporta una partida interactiva); N vectorizado en cualquier otro caso.
        n_efectivo = (1 if constants.HUMAN_OPPONENT != "none"
                        or constants.PLAY_AGAINST_AI else constants.N_BATCH)
        MainV(config, steps, N=n_efectivo).run()