"""
Snapshot de los valores por defecto de constants.py y diff contra el estado
actual del módulo. Permite que cada experimento registre la config EXACTA
(qué cambió de verdad), no solo el nombre de carpeta o un EXPERIMENT_OVERRIDES
escrito a mano que podría no haberse propagado de verdad (ver memoria: un
`from constants import X` en otro módulo deja el setattr de run_single()
sin efecto sobre ese import ya resuelto -- este diff SÍ detecta ese caso,
porque lee constants.X en el momento de llamar a compute_diff()).

IMPORTANTE -- orden de import: el snapshot se toma en el momento en que ESTE
módulo se importa por primera vez. Debe importarse (aunque sea
transitivamente) ANTES de que mainV.py aplique los overrides de config.yaml
con setattr, o el snapshot quedaría contaminado. Como este módulo solo se
importa desde experiment_log_writer.py <- multi_seed_runner.py <- mainV.py
(import de nivel superior, ANTES del bloque `if __name__`), el orden queda
garantizado sin tener que pasar el snapshot a mano por parámetro.
"""
from typing import Dict, FrozenSet, Tuple

import constants

# Dict comprehension ejecutada UNA vez, en el momento de este import.
_DEFAULTS_SNAPSHOT: Dict[str, object] = {
    name: getattr(constants, name)
    for name in dir(constants)
    if name.isupper() and not name.startswith("_")
}

# Claves que no aportan como "diff de config del experimento": son metadata
# del propio sistema de logging, o cambian por diseño DENTRO de un mismo
# experimento (SEED se muta en cada iteración del bucle de run_multi_seed;
# comparado al final del run solo reflejaría la última seed -- esa info ya
# vive en el campo "seeds" del informe, no hace falta duplicarla en el diff).
DEFAULT_DIFF_EXCLUDE: FrozenSet[str] = frozenset({
    "SEED", "MULTI_SEED_LIST", "MULTI_SEED_ENABLED", "MULTI_SEED_BASELINE",
    "RUN_NAME_SUFFIX", "VERSION",
    "EXPERIMENT_ID", "EXPERIMENT_OBJETIVO", "EXPERIMENT_HIPOTESIS",
    "EXPERIMENT_CONFUSORES", "EXPERIMENT_CONCLUSION", "EXPERIMENT_SIGUIENTE_PASO",
    "EXPERIMENT_NOTA", "EXPERIMENT_OVERRIDES", "EXPERIMENT_LOG_PATH",
    "EXPERIMENT_REPORTS_DIR", "EXPERIMENT_LOG_ENABLED",
})


def compute_diff(exclude: FrozenSet[str] = DEFAULT_DIFF_EXCLUDE) -> Dict[str, Tuple[object, object]]:
    """
    Compara el estado ACTUAL de constants.py contra el snapshot por defecto.

    Returns:
        {nombre: (valor_por_defecto, valor_actual)} solo para claves que
        difieren -- dict vacío = "este experimento corrió con la config por
        defecto, sin overrides realmente activos".
    """
    diff = {}
    for name, default_value in _DEFAULTS_SNAPSHOT.items():
        if name in exclude:
            continue
        current_value = getattr(constants, name, default_value)
        if current_value != default_value:
            diff[name] = (default_value, current_value)
    return diff