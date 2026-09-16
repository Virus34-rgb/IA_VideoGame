"""
Escritura de la fila maestra en docs/experiment_log.csv + disparo del
informe individual .md (ExperimentReportWriter). Extraído de mainV.py. Todo
el bloque queda desactivado si constants.EXPERIMENT_LOG_ENABLED=False.

Veredicto:
  - veredicto_self:   Δ winrate self-play vs baseline, umbral ±3pp.
  - veredicto_rusher: Δ medio de las 3 bandas del rusher vs baseline, ±3pp.
  - veredicto:        combinación de los dos anteriores. Si coinciden, ese
                      valor. Si difieren, "mixto (rusher=X, self=Y)". Nunca
                      se colapsa una discrepancia a un único veredicto
                      etiquetado genéricamente.

Config por seed en el .md:
  El informe individual ya NO muestra el diff vs constants.py. En su lugar
  carga el `<seed_dir>/*_config.json` que MetricsLogger.dump_config() deja
  en cada seed (idéntico entre seeds salvo SEED, típicamente).
"""
import csv
import datetime
import glob
import json
import os
import re
from typing import Dict, List, Optional

import constants
from AI.Logging.experiment_metrics import build_metrics_table
from AI.Logging.experiment_report_writer import ExperimentReportWriter
from AI.Logging.stats_parsing import load_baseline_entries


# Esquema canónico del CSV. Orden estable entre escrituras. Si añades/quitas
# columnas, actualiza también scripts/reprocess_verdicts.py.
CSV_FIELDNAMES: List[str] = [
    "id", "fecha", "objetivo", "suffix", "baseline", "overrides", "seeds", "lotes",
    "winrate_self", "delta_self_pp",
    "wr_rusher_00", "wr_rusher_05", "wr_rusher_10", "delta_rusher_pp",
    "veredicto_self", "veredicto_rusher", "veredicto",
    "pre_noisy_fix", "cross_generation",
    "nota", "artefactos", "informe_md",
]

# Umbral de decisión para los veredictos (puntos porcentuales).
_VERDICT_THRESHOLD_PP = 3.0

# Criterio de "generación post-fix NoisyLinear": IAV3+.
_POST_NOISY_FIX_GEN = 3


def _load_seed_configs(config_base_path: str, seeds: List[str]) -> Dict[str, dict]:
    """
    Carga el config.json de cada seed desde `<config_base_path>/s<seed>/`.

    Args:
        config_base_path: raíz del experimento multi-seed (contiene s42/, s43/, ...).
        seeds:            lista de labels sin prefijo 's' (p.ej. ["42", "43"]).

    Returns:
        {seed_label: {const_name: value, ...}}  con seed_label = "s42", "s43", ...
        Si una seed no tiene config.json, se omite del dict (el writer mostrará
        menos seeds en la tabla).
    """
    out: Dict[str, dict] = {}
    for seed in seeds:
        seed_label = str(seed) if str(seed).startswith("s") else f"s{seed}"
        seed_dir = os.path.join(config_base_path, seed_label)
        if not os.path.isdir(seed_dir):
            continue
        matches = glob.glob(os.path.join(seed_dir, "*_config.json"))
        if not matches:
            continue
        # Si hubiera más de uno (no debería), cogemos el más reciente.
        cfg_path = max(matches, key=os.path.getmtime)
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[exp-log][WARN] No se pudo leer {cfg_path}: {exc}")
            continue
        constants_dict = payload.get("constants", {})
        if constants_dict:
            out[seed_label] = constants_dict
    return out


def _detect_generation(path_like) -> Optional[int]:
    """Extrae 'IAV<n>' de una ruta; None si no encuentra el patrón."""
    if not path_like:
        return None
    m = re.search(r"IAV(\d+)", str(path_like))
    return int(m.group(1)) if m else None


def _verdict(delta_pp: Optional[float]) -> str:
    if delta_pp is None:
        return "N/A"
    if delta_pp >= _VERDICT_THRESHOLD_PP:
        return "positivo"
    if delta_pp <= -_VERDICT_THRESHOLD_PP:
        return "negativo"
    return "neutro"


def _combine_verdicts(v_self: str, v_rusher: str) -> str:
    """Combina los dos veredictos en uno solo, sin ocultar discrepancias."""
    if v_self == "N/A" and v_rusher == "N/A":
        return "referencia"
    if v_rusher == "N/A":
        return f"self={v_self}"
    if v_self == "N/A":
        return f"rusher={v_rusher}"
    if v_self == v_rusher:
        return v_rusher
    return f"mixto (rusher={v_rusher}, self={v_self})"


def _ensure_schema(log_path: str) -> None:
    """
    Comprueba que el CSV existente usa el esquema canónico. Si no, avisa
    sin abortar (el CSV quedará con columnas mezcladas hasta que el usuario
    lo migre con scripts/reprocess_verdicts.py o lo borre).
    """
    if not os.path.exists(log_path):
        return
    with open(log_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if header != CSV_FIELDNAMES:
        print(f"[exp-log][WARN] El CSV {log_path} tiene un esquema distinto al esperado.")
        print(f"  Esperado:   {CSV_FIELDNAMES}")
        print(f"  Encontrado: {header}")
        print("  Ejecuta scripts/reprocess_verdicts.py para migrar, o borra el CSV.")


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
    detalle completo (objetivo, hipótesis, config por seed, tabla de
    resultado completa, veredictos, confusores, conclusión). Evita duplicar
    filas con el mismo (suffix, lotes, seeds).
    """
    if not new_entries:
        return

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

    bl_entries, _ = load_baseline_entries(baseline_dir)

    # ── Delta self-play (pp) ──
    delta_self = None
    if bl_entries and win_self is not None:
        bl_self = [e["main"].get("Win ratio P1 (sin empates)") for e in bl_entries]
        bl_self = [v for v in bl_self if v is not None]
        if bl_self:
            delta_self = win_self - (sum(bl_self) / len(bl_self))

    # ── Delta rusher medio (pp, media de las 3 bandas) ──
    delta_rusher = None
    if bl_entries:
        deltas = []
        for wr_new, key in ((wr_00, "rusher_0"), (wr_05, "rusher_05"), (wr_10, "rusher_1")):
            bl_vals = [e[key].get("Victorias IA (2nd)") for e in bl_entries if key in e]
            bl_vals = [v for v in bl_vals if v is not None]
            if bl_vals and wr_new is not None:
                deltas.append((wr_new - (sum(bl_vals) / len(bl_vals))) * 100.0)
        if deltas:
            delta_rusher = sum(deltas) / len(deltas)

    veredicto_self = _verdict(delta_self)
    veredicto_rusher = _verdict(delta_rusher)
    veredicto = _combine_verdicts(veredicto_self, veredicto_rusher)

    # ── Detección de generación (pre/post fix NoisyLinear) ──
    new_gen = _detect_generation(config.base_path)
    bl_gen = _detect_generation(baseline_dir) if baseline_dir else None
    cross_generation = (new_gen is not None and bl_gen is not None and new_gen != bl_gen)
    if cross_generation:
        print(
            f"[exp-log][WARN] Comparación cruzada IAV{new_gen} vs IAV{bl_gen}. "
            f"Los runs pre-fix de NoisyLinear no son comparables directamente."
        )
    pre_noisy_fix = (new_gen is None or new_gen < _POST_NOISY_FIX_GEN)

    seeds = [e["seed"].replace("s", "") for e in new_entries]
    seeds_str = ";".join(seeds)

    # ── Autoincremento de EXP_ID si no está fijado ──
    exp_id = (constants.EXPERIMENT_ID or "").strip()
    if not exp_id:
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

    # ── Deduplicación (suffix + lotes + seeds) ──
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
    _ensure_schema(log_path)
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
        "delta_self_pp": f"{delta_self:+.2f}" if delta_self is not None else "",
        "wr_rusher_00": f"{wr_00:.4f}" if wr_00 is not None else "",
        "wr_rusher_05": f"{wr_05:.4f}" if wr_05 is not None else "",
        "wr_rusher_10": f"{wr_10:.4f}" if wr_10 is not None else "",
        "delta_rusher_pp": f"{delta_rusher:+.2f}" if delta_rusher is not None else "",
        "veredicto_self": veredicto_self,
        "veredicto_rusher": veredicto_rusher,
        "veredicto": veredicto,
        "pre_noisy_fix": pre_noisy_fix,
        "cross_generation": cross_generation,
        "nota": constants.EXPERIMENT_NOTA,
        "artefactos": artefactos,
        "informe_md": informe_md,
    }

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[exp-log] Añadida fila {exp_id} a {log_path}")

    # ── Cargar config.json de cada seed para el informe .md ──
    seed_configs = _load_seed_configs(config.base_path, seeds)

    # Tabla de métricas + métricas usadas (misma fuente que el comparador de consola)
    col_headers, table_rows, key_metrics_used = build_metrics_table(new_entries, bl_entries)

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
        seed_configs=seed_configs,
        col_headers=col_headers,
        table_rows=table_rows,
        key_metrics=key_metrics_used,
        confusores=constants.EXPERIMENT_CONFUSORES,
        conclusion=constants.EXPERIMENT_CONCLUSION,
        siguiente_paso=constants.EXPERIMENT_SIGUIENTE_PASO,
        veredicto=veredicto,
        veredicto_self=veredicto_self,
        veredicto_rusher=veredicto_rusher,
        delta_self_pp=delta_self,
        delta_rusher_pp=delta_rusher,
        pre_noisy_fix=pre_noisy_fix,
        cross_generation=cross_generation,
        nota=constants.EXPERIMENT_NOTA,
        artefactos_str=artefactos,
    )
    print(f"[exp-log] Informe individual guardado en {informe_md}")