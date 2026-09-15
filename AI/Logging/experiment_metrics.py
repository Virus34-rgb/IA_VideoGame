"""
Tabla de métricas compartida entre el comparador de consola
(experiment_comparator.compare_seeds_against_baseline) y el informe .md
individual (experiment_log_writer). Única fuente de verdad de qué métricas
se comparan y cómo se calculan sus medias/deltas.
"""
from typing import List, Optional, Tuple

# ("sección del entry", "nombre exacto de la métrica en el stats.txt", "formato")
#
# FIX: la sección "rusher_X" solo tiene la clave "Victorias IA (2nd)" (el
# RATIO, ej. 0.6098) -- "Victorias IA" a secas es el CONTEO (ej. 12490).
# La versión anterior leía "Victorias IA vs rusher 0.0", que no existe en
# ningún stats.txt real (StatsReportWriter._section_rusher solo escribe
# "Victorias IA"), así que esas 3 filas siempre mostraban N/A.
EXPERIMENT_KEY_METRICS: List[Tuple[str, str, str]] = [
    ("main",      "Win ratio P1 (sin empates)",      "{:.2f}"),
    ("main",      "Turnos medios por partida",       "{:.2f}"),
    ("main",      "Elo P1",                          "{:.1f}"),
    ("main",      "Elo P2",                          "{:.1f}"),
    ("main",      "Daño medio P1",                   "{:.2f}"),
    ("main",      "Reward media P1",                 "{:.2f}"),
    ("main",      "Bajas medias P1",                 "{:.2f}"),
    ("main",      "Daño por sobrekill medio P1",     "{:.2f}"),
    ("main",      "Kill confirmed medio P1",         "{:.2f}"),
    ("main",      "Defensas desperdiciadas P1",      "{:.2f}"),
    ("main",      "Movimientos estratégicos P1 (%)", "{:.2f}"),
    ("rusher_0",  "Victorias IA (2nd)",              "{:.4f}"),
    ("rusher_05", "Victorias IA (2nd)",              "{:.4f}"),
    ("rusher_1",  "Victorias IA (2nd)",              "{:.4f}"),
]


def build_metrics_table(
    new_entries: list,
    baseline_entries: list,
    key_metrics: Optional[List[Tuple[str, str, str]]] = None,
) -> Tuple[List[str], List[List[str]], List[Tuple[str, str, str]]]:
    """
    Construye la matriz BL_* | BL_μ | NW_* | NW_μ | Δμ.

    Returns:
        col_headers: nombres de columna en orden.
        table_rows: una fila (strings ya formateados) por métrica.
        key_metrics: eco del argumento, o EXPERIMENT_KEY_METRICS por defecto.
    """
    if key_metrics is None:
        key_metrics = EXPERIMENT_KEY_METRICS

    n_bl = len(baseline_entries)
    n_nw = len(new_entries)
    # Medias (BL_μ/NW_μ) y delta solo tienen sentido estadístico con >=2
    # seeds -- con 1 sola seed "media" == el propio valor, columna redundante.
    show_bl_mean = n_bl >= 2
    show_nw_mean = n_nw >= 2
    show_delta = show_bl_mean and show_nw_mean

    col_headers: List[str] = [f"BL_{e['seed']}" for e in baseline_entries]
    if show_bl_mean:
        col_headers.append("BL_μ")
    col_headers += [f"NW_{e['seed']}" for e in new_entries]
    if show_nw_mean:
        col_headers.append("NW_μ")
    if show_delta:
        col_headers.append("Δμ")

    def _fmt(value, fmt):
        return fmt.format(value) if value is not None else "N/A"

    def _get(entry, section, name):
        src = entry["main"] if section == "main" else entry[section]
        return src.get(name)

    table_rows: List[List[str]] = []
    for section, name, fmt in key_metrics:
        row: List[str] = []

        bl_vals = [_get(e, section, name) for e in baseline_entries]
        row += [_fmt(v, fmt) for v in bl_vals]
        if show_bl_mean:
            nums = [v for v in bl_vals if v is not None]
            row.append(_fmt(sum(nums) / len(nums) if nums else None, fmt))

        nw_vals = [_get(e, section, name) for e in new_entries]
        row += [_fmt(v, fmt) for v in nw_vals]
        if show_nw_mean:
            nums = [v for v in nw_vals if v is not None]
            row.append(_fmt(sum(nums) / len(nums) if nums else None, fmt))

        if show_delta:
            bl_nums = [v for v in bl_vals if v is not None]
            nw_nums = [v for v in nw_vals if v is not None]
            if bl_nums and nw_nums:
                delta = sum(nw_nums) / len(nw_nums) - sum(bl_nums) / len(bl_nums)
                row.append(fmt.format(delta))
            else:
                row.append("N/A")

        table_rows.append(row)

    return col_headers, table_rows, key_metrics