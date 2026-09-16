"""
Tabla de métricas compartida entre el comparador de consola
(experiment_comparator.compare_seeds_against_baseline) y el informe .md
individual (experiment_log_writer). Única fuente de verdad de qué métricas
se comparan y cómo se calculan sus medias/deltas.

Esquema de key_metrics:
    (section, key, display_name, fmt, scale)

  - section:      clave del dict de entry ("main", "rusher_0", "rusher_05", "rusher_1").
  - key:          nombre exacto de la métrica en el stats.txt.
  - display_name: nombre legible que se muestra en el informe .md y en la
                  tabla ASCII del comparador de consola.
  - fmt:          formato Python ("{:.2f}", "{:.2f}%", ...).
  - scale:        factor multiplicativo aplicado ANTES de formatear. Se usa
                  para pasar ratios (0-1) a porcentaje (scale=100.0), y para
                  que la columna Δμ muestre la diferencia en puntos
                  porcentuales con el mismo factor.
"""
from typing import List, Optional, Tuple

# (section, key, display_name, fmt, scale)
EXPERIMENT_KEY_METRICS: List[Tuple[str, str, str, str, float]] = [
    ("main",      "Win ratio P1 (sin empates)",       "Win ratio P1 (sin empates)",       "{:.2f}",  1.0),
    ("main",      "Turnos medios por partida",        "Turnos medios por partida",        "{:.2f}",  1.0),
    ("main",      "Elo P1",                           "Elo P1",                           "{:.1f}",  1.0),
    ("main",      "Elo P2",                           "Elo P2",                           "{:.1f}",  1.0),
    ("main",      "Daño medio P1",                    "Daño medio P1",                    "{:.2f}",  1.0),
    ("main",      "Reward media P1",                  "Reward media P1",                  "{:.2f}",  1.0),
    ("main",      "Bajas medias P1",                  "Bajas medias P1",                  "{:.2f}",  1.0),
    ("main",      "Daño por sobrekill medio P1",      "Daño por sobrekill medio P1",      "{:.2f}",  1.0),
    ("main",      "Kill confirmed medio P1",          "Kill confirmed medio P1",          "{:.2f}",  1.0),
    ("main",      "Defensas desperdiciadas P1",       "Defensas desperdiciadas P1",       "{:.2f}",  1.0),
    ("main",      "Movimientos estratégicos P1 (%)",  "Movimientos estratégicos P1 (%)",  "{:.2f}",  1.0),
    # Los tests vs Rusher se corren con agresividad fija 0.0 / 0.5 / 1.0
    # (ver step_builder.py: rusher_aggression=0.0 / 0.5 / 1.0). El display_name
    # refleja los bandos de agresividad del dataset de test.
    ("rusher_0",  "Victorias IA (2nd)",               "Winrate vs Rusher (agg 0.0–0.3)",  "{:.2f}%", 100.0),
    ("rusher_05", "Victorias IA (2nd)",               "Winrate vs Rusher (agg 0.3–0.6)",  "{:.2f}%", 100.0),
    ("rusher_1",  "Victorias IA (2nd)",               "Winrate vs Rusher (agg 0.6–1.0)",  "{:.2f}%", 100.0),
]


def build_metrics_table(
    new_entries: list,
    baseline_entries: list,
    key_metrics: Optional[List[Tuple[str, str, str, str, float]]] = None,
) -> Tuple[List[str], List[List[str]], List[Tuple[str, str, str, str, float]]]:
    """
    Construye la matriz BL_* | BL_μ | NW_* | NW_μ | Δμ.

    Args:
        new_entries:      entries cargados de cada seed nueva.
        baseline_entries: entries cargados del baseline (0 o más).
        key_metrics:      lista opcional de tuplas; si None, usa EXPERIMENT_KEY_METRICS.

    Returns:
        col_headers: nombres de columna en orden.
        table_rows:  una fila (strings ya formateados) por métrica.
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

    def _fmt(value, fmt: str, scale: float = 1.0) -> str:
        """Aplica scale antes de formatear. None → 'N/A'."""
        if value is None:
            return "N/A"
        try:
            return fmt.format(value * scale)
        except (TypeError, ValueError):
            return "N/A"

    def _get(entry, section: str, name: str):
        src = entry["main"] if section == "main" else entry[section]
        return src.get(name)

    table_rows: List[List[str]] = []
    for section, key, _display, fmt, scale in key_metrics:
        row: List[str] = []

        bl_vals = [_get(e, section, key) for e in baseline_entries]
        row += [_fmt(v, fmt, scale) for v in bl_vals]
        if show_bl_mean:
            nums = [v for v in bl_vals if v is not None]
            row.append(_fmt(sum(nums) / len(nums) if nums else None, fmt, scale))

        nw_vals = [_get(e, section, key) for e in new_entries]
        row += [_fmt(v, fmt, scale) for v in nw_vals]
        if show_nw_mean:
            nums = [v for v in nw_vals if v is not None]
            row.append(_fmt(sum(nums) / len(nums) if nums else None, fmt, scale))

        if show_delta:
            bl_nums = [v for v in bl_vals if v is not None]
            nw_nums = [v for v in nw_vals if v is not None]
            if bl_nums and nw_nums:
                delta = sum(nw_nums) / len(nw_nums) - sum(bl_nums) / len(bl_nums)
                # Mismo scale que los valores: para winrate vs Rusher, el delta
                # sale directamente en puntos porcentuales (ej. "+5.00%").
                row.append(_fmt(delta, fmt, scale))
            else:
                row.append("N/A")

        table_rows.append(row)

    return col_headers, table_rows, key_metrics