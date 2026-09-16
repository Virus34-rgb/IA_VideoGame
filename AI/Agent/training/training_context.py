"""
Contexto de configuración explícito para las piezas sensibles a overrides
en comparaciones (RunSpec). Sustituye la lectura directa de constants.X
dentro de los colaboradores extraídos por inyección explícita.

Uso: se construye UNA vez por run (en run_single(), DESPUÉS de aplicar los
overrides de RunSpec sobre el módulo constants) y se propaga explícitamente
hasta los colaboradores que lo necesitan. constants.py sigue siendo la
fuente de valores por defecto cuando no se pasa contexto (compatibilidad
hacia atrás).

Nota de diseño:
  - dataclass(frozen=True) → los campos no se pueden reasignar tras
    construcción (categoría C: inmutable tras construcción).
  - rusher_aggression_band_weights se almacena como tuple (no list) porque
    frozen=True solo congela la referencia del campo, no el contenido mutable
    de una list. Con tuple, el objeto es realmente inmutable.
  - Se distinguen copy_dqn_sel y copy_dqn_turn (en lugar de un único
    copy_dqn): constants.py tiene COPY_DQN_SEL y COPY_DQN_TURN como
    constantes separadas desde la limpieza del dead COPY_DQN genérico.
  - No incluye BETA_DECAY_RATE: vive dentro de ReplayMemoryAN/ReplayMemoryPM
    (fuera del alcance de esta fase). Paso de seguimiento opcional.
"""
from dataclasses import dataclass
from typing import Tuple

import constants as _constants_module


@dataclass(frozen=True)
class TrainingContext:
    batch_size: int
    turn_replays_per_batch: int
    selection_replays_per_batch: int
    rusher_aggression_band_weights: Tuple[float, ...]
    copy_dqn_sel: int
    copy_dqn_turn: int

    @classmethod
    def from_constants(cls) -> "TrainingContext":
        """Snapshot de los valores actuales de constants -- llamar DESPUÉS de
        aplicar cualquier override de RunSpec sobre el módulo, para capturar
        los valores efectivos de ESTE run concreto."""
        return cls(
            batch_size=_constants_module.BATCH_SIZE,
            turn_replays_per_batch=_constants_module.TURN_REPLAYS_PER_BATCH,
            selection_replays_per_batch=_constants_module.SELECTION_REPLAYS_PER_BATCH,
            rusher_aggression_band_weights=tuple(_constants_module.RUSHER_AGGRESSION_BAND_WEIGHTS),
            copy_dqn_sel=_constants_module.COPY_DQN_SEL,
            copy_dqn_turn=_constants_module.COPY_DQN_TURN,
        )