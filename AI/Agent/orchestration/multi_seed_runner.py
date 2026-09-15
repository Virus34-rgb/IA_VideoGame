"""
Orquestación del modo multi-seed: corre el pipeline completo (MainV.run())
una vez por cada seed en una subcarpeta 's<seed>/', y al final genera el
reporte comparativo + la fila en experiment_log.csv. Extraído de mainV.py.
"""
import os
import re
import time
from typing import List

import constants
from AI.Agent.orchestration.main_runner import MainV
from AI.Agent.orchestration.seed_utils import set_seed
from AI.Agent.orchestration.step_builder import build_steps
from AI.Logging.experiment_comparator import compare_seeds_against_baseline
from AI.Logging.experiment_log_writer import append_experiment_log_entry
from AI.Logging.stats_parsing import find_seed_dirs, load_stats_entry
from config import RunConfig


def run_multi_seed(base_config: RunConfig, seeds: List[int], baseline_dir=None) -> None:
    """
    Ejecuta el pipeline completo para cada seed en `seeds`. Cada seed se
    guarda en una subcarpeta 's<seed>/' DENTRO de la carpeta base:

        models/IAV{version}_{suffix}/
        ├── s42/  (P1/  P2/  stats.txt  stats2.txt  stats_rusher_*.txt ...)
        ├── s43/
        ├── s44/
        ├── comparison_report.txt
        └── comparison_report.json
    """
    base_root = base_config.base_path
    os.makedirs(base_root, exist_ok=True)

    print("=" * 70)
    print(f"MODO MULTI-SEED  |  seeds = {seeds}")
    print(f"Directorio raíz: {base_root}")
    print("=" * 70)

    for seed in seeds:
        seed_dir = os.path.join(base_root, f"s{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        # base_path_override fuerza que TODAS las rutas derivadas de RunConfig
        # (p1_path, stats_path, path_opp_pool, ...) apunten a la subcarpeta
        # de ESTA seed, sin tocar ningún otro punto del pipeline.
        seed_config = RunConfig(
            version=base_config.version,
            train_episodes=base_config.train_episodes,
            eval_episodes=base_config.eval_episodes,
            suffix=base_config.suffix,
            base_dir=base_config.base_dir,
            base_path_override=seed_dir,
        )

        print(f"\n{'#' * 65}\n# SEED {seed} → {seed_dir}\n{'#' * 65}")

        constants.SEED = seed  # por si algún módulo lo lee directamente
        set_seed(seed)

        steps = build_steps(seed_config)
        # N=1 obligatorio en modos con jugador humano (PlayerNoAIV solo
        # soporta una partida interactiva); N vectorizado en cualquier otro caso.
        n_efectivo = (1 if constants.HUMAN_OPPONENT != "none"
                        or constants.PLAY_AGAINST_AI else constants.N_BATCH)

        t0 = time.time()
        MainV(seed_config, steps, N=n_efectivo).run()
        elapsed = time.time() - t0
        print(f"[seed {seed}] terminado en {MainV._format_time(elapsed)}")

    compare_seeds_against_baseline(
        seeds_dir=base_root,
        baseline_dir=baseline_dir,
        output_txt=os.path.join(base_root, "comparison_report.txt"),
        output_json=os.path.join(base_root, "comparison_report.json"),
        suffix_filter=None,   # layout nuevo: se buscan subcarpetas 'sN' directas
    )
    print(f"[compare] Reporte guardado en {os.path.join(base_root, 'comparison_report.txt')}")

    # Recarga las entries recién escritas desde disco (en vez de reutilizar
    # las del bucle de arriba), para que el log maestro lea EXACTAMENTE lo
    # mismo que acaba de imprimir compare_seeds_against_baseline -- una única
    # fuente de verdad en disco.
    new_entries = []
    for sd in find_seed_dirs(base_root, suffix_filter=None):
        name = os.path.basename(sd)
        m = re.search(r"_s(\d+)$", name) or re.fullmatch(r"s(\d+)", name)
        short = f"s{m.group(1)}" if m else name
        new_entries.append(load_stats_entry(sd, short))

    append_experiment_log_entry(
        log_path=constants.EXPERIMENT_LOG_PATH,
        new_entries=new_entries,
        config=base_config,
        overrides=constants.EXPERIMENT_OVERRIDES if hasattr(constants, "EXPERIMENT_OVERRIDES") else None,
        baseline_dir=baseline_dir,
    )