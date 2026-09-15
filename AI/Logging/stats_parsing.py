"""
Lectura de archivos stats*.txt y localización de carpetas de seeds
(IAV*_s<seed>/ o s<seed>/) en disco. Extraído de mainV.py -- funciones puras
de I/O + parsing, sin lógica de comparación ni de logging.
"""
import glob
import os
import re
from typing import List, Optional, Tuple


def parse_stats_file(path: str) -> dict:
    """
    Extrae {nombre_metrica: valor_numerico} de un archivo stats.txt.
    Acepta líneas tipo:
        "Partidas:                  40960"
        "Victorias P1:              26911 (65.70%)"
        "Victorias IA:              12490 -> 0.6098"
    """
    metrics = {}
    if not os.path.exists(path):
        return metrics
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip()
            if ":" not in line:
                continue
            name, rest = line.split(":", 1)
            name = name.strip()
            if not name or name.startswith("-") or name.startswith("="):
                continue
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?", rest)
            if not nums:
                continue
            try:
                vals = [float(x) for x in nums]
            except ValueError:
                continue
            metrics[name] = vals[0]
            if len(vals) >= 2:
                # Segunda cifra: cubre TANTO "12490 -> 0.6098" (ratio) COMO
                # "26911 (65.70%)" (porcentaje) bajo una misma clave "(2nd)",
                # para que el resto del pipeline no distinga el formato de origen.
                metrics[f"{name} (2nd)"] = vals[1]
    return metrics


def find_seed_dirs(parent_dir: str, suffix_filter: Optional[str] = None) -> List[str]:
    """
    Devuelve las subcarpetas de `parent_dir` que corresponden a una seed.
    Reconoce dos layouts:
      1. Nuevo: 'parent_dir/s42/', 'parent_dir/s43/', ...  (subcarpetas directas)
      2. Antiguo: 'parent_dir/IAV*_s42/', '..._s43/', ... (carpetas hermanas)
    Si `suffix_filter` se pasa, solo aplica al layout 2.
    """
    candidates: List[str] = []
    candidates += glob.glob(os.path.join(parent_dir, "s[0-9]*"))

    if suffix_filter:
        pattern = os.path.join(parent_dir, f"IAV*_{suffix_filter}_s[0-9]*")
    else:
        pattern = os.path.join(parent_dir, "IAV*_s[0-9]*")
    candidates += glob.glob(pattern)

    valid = []
    for p in candidates:
        base = os.path.basename(os.path.normpath(p))
        if re.fullmatch(r"s\d+", base) or re.search(r"_s\d+$", base):
            valid.append(p)

    def _extract_seed(p):
        base = os.path.basename(os.path.normpath(p))
        m = re.fullmatch(r"s(\d+)", base) or re.search(r"_s(\d+)$", base)
        return int(m.group(1)) if m else 0

    seen = set()
    unique = []
    for p in valid:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return sorted(unique, key=_extract_seed)


def load_stats_entry(run_dir: str, label: str) -> dict:
    """
    Carga stats2.txt y stats_rusher_*.txt de `run_dir`. `label` es la
    etiqueta corta para columna (p. ej. 's42', 'base', 'b1').
    """
    entry = {"seed": label, "main": {}, "rusher_0": {}, "rusher_05": {}, "rusher_1": {}}
    for fname in ("stats2.txt", "stats.txt"):
        p = os.path.join(run_dir, fname)
        if os.path.exists(p):
            entry["main"] = parse_stats_file(p)
            break
    for tag, fname in (("rusher_0", "stats_rusher_aggr_0.txt"),
                       ("rusher_05", "stats_rusher_aggr_05.txt"),
                       ("rusher_1", "stats_rusher_aggr_1.txt")):
        p = os.path.join(run_dir, fname)
        if os.path.exists(p):
            entry[tag] = parse_stats_file(p)
    return entry


def load_baseline_entries(baseline_dir) -> Tuple[list, bool]:
    """
    Devuelve (lista_de_entries, es_multi). Acepta:
      - None → ([], False)
      - str a un fichero de stats → un único entry
      - str a una carpeta con stats2.txt → un único entry
      - str a una carpeta con subcarpetas IAV*_s<seed> → un entry por subcarpeta
      - list/tuple de str, cada uno un run distinto → un entry por elemento
    """
    if baseline_dir is None:
        return [], False

    if isinstance(baseline_dir, (list, tuple)):
        entries = []
        for i, path in enumerate(baseline_dir):
            if not os.path.exists(path):
                print(f"[compare] Baseline path no existe: {path}")
                continue
            name = os.path.basename(os.path.normpath(path))
            m = re.search(r"_s(\d+)$", name)
            label = f"s{m.group(1)}" if m else f"b{i+1}"
            entries.append(load_stats_entry(path, label))
        return entries, len(entries) > 1

    if not os.path.exists(baseline_dir):
        print(f"[compare] Baseline no existe: {baseline_dir}")
        return [], False

    if os.path.isfile(baseline_dir):
        entry = {"seed": "base", "main": parse_stats_file(baseline_dir),
                 "rusher_0": {}, "rusher_05": {}, "rusher_1": {}}
        return [entry], False

    seed_subdirs = find_seed_dirs(baseline_dir)
    if seed_subdirs:
        entries = []
        for sd in seed_subdirs:
            name = os.path.basename(sd)
            m = re.fullmatch(r"s(\d+)", name) or re.search(r"_s(\d+)$", name)
            short = f"s{m.group(1)}" if m else name
            entries.append(load_stats_entry(sd, short))
        return entries, True

    entry = load_stats_entry(baseline_dir, "base")
    return [entry], False