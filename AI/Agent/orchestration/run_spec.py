"""
Modo comparación de runs (RunSpec): ejecuta varias configuraciones con
overrides de constants.py distintos y superpone sus curvas de progreso.
Extraído de mainV.py.

TrainingContext:
  run_single() construye un TrainingContext DESPUÉS de aplicar los overrides
  de RunSpec sobre el módulo constants, y lo propaga a MainV. Esto garantiza
  que cualquier colaborador que lea BATCH_SIZE/COPY_DQN/etc. a través del
  context reciba los valores EFECTIVOS de este run concreto, no los del
  momento de import del módulo (que es lo que rompía el patrón
  `from constants import X`).
"""
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import yaml

import constants
from AI.Agent.orchestration.main_runner import MainV
from AI.Agent.orchestration.seed_utils import set_seed
from AI.Agent.orchestration.step_builder import build_steps
from AI.Agent.training.training_context import TrainingContext
from AI.Logging.metrics_logger import MetricsLogger
from config import RunConfig


@dataclass
class RunSpec:
    """Especificación de un run para comparación (nombre, N, lotes, overrides
    de constants.py, clase de jugador, seed)."""
    run_name: str
    N: int
    train_batches: int
    eval_batches: int
    constants_overrides: dict = field(default_factory=dict)
    player_class: Optional[Callable] = None
    seed: Optional[int] = None


def load_config_yaml(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        print(f"Advertencia: {path} no encontrado. Usando valores por defecto.")
        return {}
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_single(
    config: RunConfig,
    run_spec: RunSpec,
    shared_log_dir: Optional[str] = None,
) -> tuple[str, str]:
    """Ejecuta un único run con la configuración y especificaciones dadas."""
    if run_spec.seed is not None and constants.SEED is not None:
        set_seed(run_spec.seed)

    # setattr temporal sobre el módulo constants: el pipeline entero lee sus
    # valores vía `import constants; constants.X`, así que esto SÍ se
    # propaga -- SALVO en los sitios que hagan `from constants import X`
    # (ese patrón queda inmune al setattr porque ya capturó el valor en el
    # momento del import).
    #
    # TrainingContext cierra esa brecha estructuralmente para las 5 constantes
    # cubiertas: se snapshotea DESPUÉS de aplicar overrides y se inyecta
    # explícitamente en los colaboradores. Los sitios que aún lean
    # `from constants import X` de otras constantes siguen siendo
    # responsabilidad de AI.Logging.constants_diff.compute_diff().
    original_values = {}
    for key, value in run_spec.constants_overrides.items():
        original_values[key] = getattr(constants, key)
        setattr(constants, key, value)

    try:
        # Snapshot de constants DESPUÉS de aplicar los overrides de este
        # run concreto -- captura los valores EFECTIVOS de esta comparación.
        context = TrainingContext.from_constants()

        run_config = RunConfig(
            version=f"{config.version}_{run_spec.run_name}",
            train_episodes=run_spec.train_batches,
            eval_episodes=run_spec.eval_batches,
        )
        steps = build_steps(run_config)

        main = MainV(
            run_config,
            steps,
            N=run_spec.N,
            player_class=run_spec.player_class,
            log_dir=shared_log_dir,
            context=context,
        )
        main.run()
        return main.log_dir, f"v{run_config.version}"
    finally:
        # Restaurar SIEMPRE (incluso si main.run() lanza), para que un run
        # fallido no deje overrides "pegados" al siguiente RunSpec de la lista.
        for key, value in original_values.items():
            setattr(constants, key, value)


def run_comparison(config: RunConfig, run_specs: List[RunSpec]) -> None:
    shared_log_dir = config.base_path
    run_names = []

    for spec in run_specs:
        print(f"\n{'#' * 65}\n RUN: {spec.run_name}\n{'#' * 65}")
        _, versioned_name = run_single(config, spec, shared_log_dir=shared_log_dir)
        run_names.append(versioned_name)

    MetricsLogger.compare_runs(
        shared_log_dir,
        run_names,
        labels=[s.run_name for s in run_specs],
        show=False,
    )