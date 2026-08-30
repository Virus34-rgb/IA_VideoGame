"""
Módulo de configuración y logging para Weights & Biases (wandb).
Centraliza la inicialización y el envío de métricas.
"""
import wandb
from typing import Any, Dict, Optional

import constants


def init_wandb(
    project_name: str = "castle-game-rl",
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Inicializa una ejecución de wandb.

    Args:
        project_name: Nombre del proyecto en wandb.
        run_name: Nombre de la ejecución (se usa como identificador).
        config: Diccionario con los hiperparámetros a registrar.
    """
    if not constants.USE_WANDB:
        return

    if config is None:
        config = {}

    # Si no se proporciona run_name, se genera automáticamente con la versión
    if run_name is None:
        run_name = f"v{constants.VERSION}"

    wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        reinit=True,  # Permite reiniciar si ya existe una ejecución (útil en bucles)
    )


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """
    Envía métricas a wandb de forma asíncrona.

    Args:
        metrics: Diccionario con las métricas a registrar.
        step: Número de paso (batch/época) para asociar a las métricas.
    """
    if not constants.USE_WANDB:
        return
    if step is not None:
        metrics["global_step"] = step
    wandb.log(metrics)


def finish_wandb() -> None:
    """Finaliza la ejecución de wandb (opcional, se llama al salir)."""
    if constants.USE_WANDB:
        wandb.finish()