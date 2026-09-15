"""
Escritura de la fila maestra en docs/experiment_log.csv + disparo del
informe individual .md (ExperimentReportWriter). Extraído de mainV.py. Todo
el bloque queda desactivado si constants.EXPERIMENT_LOG_ENABLED=False.
"""
import csv
import datetime
import os
import re
from typing import Optional

import constants
from AI.Logging.constants_diff import compute_diff
from AI.Logging.experiment_metrics import build_metrics_table
from AI.Logging.experiment_report_writer import ExperimentReportWriter
from AI.Logging.stats_parsing import load_baseline_entries


def append_experiment_log_entry(
    log_path: str,
    new_entries: list,
    config,
    overrides: Optional[dict] = None,
    baseline_dir=None,
) -> None:
    """
    Añade una fila a docs/experiment_log.csv con las medias de las seeds
    recién corridas, y genera docs/experiments/<EXP_ID>_<suffix>.md con el
    detalle completo (objetivo, hipótesis, diff de config, tabla de
    resultado completa, confusores, conclusión). Evita duplicar filas con
    el mismo (suffix, lotes, seeds).
    """
    if not new_entries:
        return

    # Flag maestro: con esto en False, ni el CSV ni el informe .md se tocan,
    # sin importar si el resto del pipeline (multi-seed, comparación) corrió.
    if not constants.EXPERIMENT_LOG_ENABLED:
        print("[exp-log] EXPERIMENT_LOG_ENABLED=False -- no se escribe CSV ni informe .md.")
        return

    def _mean(key, section="main"):
        vals = [e.get(section, {}).get(key) for e in new_entries]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    win_self = _mean("Win ratio P1 (sin empates)")
    wr_00 = _mean("Victorias IA (2nd)", "rusher_0")
    wr_05 = _mean("Victorias IA (2nd)", "rusher_05")
    wr_10 = _mean("Victorias IA (2nd)", "rusher_1")

    # Baseline se carga UNA vez y se reutiliza tanto para el veredicto
    # cualitativo como para la tabla completa del informe .md.
    bl_entries, _ = load_baseline_entries(baseline_dir)

    # Veredicto derivado: comparamos solo self-play porque es la métrica más
    # ruidosa y la más correlacionada con "vale la pena adoptar esto" (ver
    # memoria: winrate P1-vs-P2 es la señal más ruidosa por dinámica self-play).
    veredicto = "referencia"
    if bl_entries:
        bl_self = [e["main"].get("Win ratio P1 (sin empates)") for e in bl_entries]
        bl_self = [v for v in bl_self if v is not None]
        if bl_self and win_self is not None:
            delta = win_self - (sum(bl_self) / len(bl_self))
            if delta >= 3.0:
                veredicto = "positivo"
            elif delta <= -3.0:
                veredicto = "negativo"
            else:
                veredicto = "neutro"

    seeds = [e["seed"].replace("s", "") for e in new_entries]
    seeds_str = ";".join(seeds)

    exp_id = (constants.EXPERIMENT_ID or "").strip()
    if not exp_id:
        # Autoincremento: busca el mayor "EXP-000N" ya escrito en el CSV y
        # continúa desde ahí; si el CSV no existe, arranca en EXP-0001.
        next_n = 1
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)  # header
                for row in reader:
                    if row and row[0].startswith("EXP-"):
                        try:
                            n = int(row[0].split("-")[1])
                            next_n = max(next_n, n + 1)
                        except (IndexError, ValueError):
                            pass
        exp_id = f"EXP-{next_n:04d}"

    # Deduplicación simple: mismo suffix + lotes + seeds -> no añadir de
    # nuevo (evita filas duplicadas al re-correr el mismo experimento).
    dedup_key = (config.suffix, config.train_episodes, seeds_str)
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if (row["suffix"], int(row["lotes"]), row["seeds"]) == dedup_key:
                        print(f"[exp-log] Ya existe fila para {dedup_key}, se omite.")
                        return
                except (KeyError, ValueError):
                    continue

    ov_str = ";".join(f"{k}={v}" for k, v in overrides.items()) if overrides else ""

    if baseline_dir is None:
        bl_str = ""
    elif isinstance(baseline_dir, (list, tuple)):
        bl_str = ";".join(baseline_dir)
    else:
        bl_str = str(baseline_dir)

    artefactos = ";".join(os.path.join(config.base_path, f"s{s}") for s in seeds)

    clean_suffix = re.sub(r'[^a-zA-Z0-9_\-]', '_', config.suffix) if config.suffix else exp_id
    informe_md = os.path.join(constants.EXPERIMENT_REPORTS_DIR, f"{exp_id}_{clean_suffix}.md")

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    write_header = not os.path.exists(log_path)

    row = {
        "id": exp_id,
        "fecha": datetime.date.today().isoformat(),
        "objetivo": constants.EXPERIMENT_OBJETIVO,
        "suffix": config.suffix,
        "baseline": bl_str,
        "overrides": ov_str,
        "seeds": seeds_str,
        "lotes": config.train_episodes,
        "winrate_self": f"{win_self:.2f}" if win_self is not None else "",
        "wr_rusher_00": f"{wr_00:.4f}" if wr_00 is not None else "",
        "wr_rusher_05": f"{wr_05:.4f}" if wr_05 is not None else "",
        "wr_rusher_10": f"{wr_10:.4f}" if wr_10 is not None else "",
        "veredicto": veredicto,
        "nota": constants.EXPERIMENT_NOTA,
        "artefactos": artefactos,
        "informe_md": informe_md,
    }

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[exp-log] Añadida fila {exp_id} a {log_path}")

    # Tabla completa (no solo el Δ final) + diff real de constants.py, para
    # el informe .md -- misma fuente de datos que el comparador de consola
    # (build_metrics_table), así que nunca pueden divergir entre sí.
    col_headers, table_rows, key_metrics_used = build_metrics_table(new_entries, bl_entries)
    constants_diff = compute_diff()

    ExperimentReportWriter.write(
        path=informe_md,
        exp_id=exp_id,
        fecha=row["fecha"],
        objetivo=constants.EXPERIMENT_OBJETIVO,
        hipotesis=constants.EXPERIMENT_HIPOTESIS,
        suffix=config.suffix,
        seeds=seeds,
        lotes=config.train_episodes,
        n_batch=constants.N_BATCH,
        baseline_str=bl_str,
        constants_diff=constants_diff,
        col_headers=col_headers,
        table_rows=table_rows,
        key_metrics=key_metrics_used,
        confusores=constants.EXPERIMENT_CONFUSORES,
        conclusion=constants.EXPERIMENT_CONCLUSION,
        siguiente_paso=constants.EXPERIMENT_SIGUIENTE_PASO,
        veredicto=veredicto,
        nota=constants.EXPERIMENT_NOTA,
        artefactos_str=artefactos,
    )
    print(f"[exp-log] Informe individual guardado en {informe_md}")