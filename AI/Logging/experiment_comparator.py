"""
Comparador de consola: tabla ASCII BL_* vs NW_* para N seeds nuevas contra
uno o varios baselines. Extraído de mainV.py -- solo construye e
imprime/guarda el reporte de texto; NO escribe en experiment_log.csv (eso es
responsabilidad de experiment_log_writer.append_experiment_log_entry).
"""
import json
import os
import re
from typing import List, Optional

from AI.Logging.experiment_metrics import build_metrics_table
from AI.Logging.stats_parsing import find_seed_dirs, load_baseline_entries, load_stats_entry


def compare_seeds_against_baseline(
    seeds_dir: str,
    baseline_dir=None,
    output_json: Optional[str] = None,
    output_txt: Optional[str] = None,
    suffix_filter: Optional[str] = None,
) -> str:
    """
    Compara stats por seed contra uno o varios baselines.

    Layout de columnas:
      Métrica | BL_s42 | BL_s43 | BL_s44 | BL_μ | NW_s42 | NW_s43 | NW_s44 | NW_μ | Δμ
    BL_μ/NW_μ solo aparecen con >=2 seeds en su grupo; Δμ = NW_μ - BL_μ solo
    si ambos grupos tienen media.
    """
    new_entries = []
    for sd in find_seed_dirs(seeds_dir, suffix_filter=suffix_filter):
        name = os.path.basename(sd)
        m = re.search(r"_s(\d+)$", name)
        short = f"s{m.group(1)}" if m else name
        new_entries.append(load_stats_entry(sd, short))

    if not new_entries:
        msg = f"[compare] No se encontraron subcarpetas IAV*_s<seed> en {seeds_dir}"
        if output_txt:
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write(msg + "\n")
        else:
            print(msg)
        return msg

    baseline_entries, _baseline_is_multi = load_baseline_entries(baseline_dir)

    col_headers, table_rows, key_metrics = build_metrics_table(new_entries, baseline_entries)
    n_bl = len(baseline_entries)
    n_nw = len(new_entries)
    n_val_cols = len(col_headers)

    # Anchos dinámicos: cada columna se ensancha hasta su contenido más
    # largo (cabecera o valor) + 2 de margen, para que la tabla ASCII no
    # quede desalineada con métricas/etiquetas de longitud muy distinta.
    metric_width = max(
        len("Métrica"),
        max((len(name) for _, name, _ in key_metrics), default=0),
    ) + 2
    value_widths = []
    for j, h in enumerate(col_headers):
        content_max = max((len(r[j]) for r in table_rows), default=0)
        value_widths.append(max(len(h), content_max) + 2)

    lines: List[str] = []

    def _border(left, mid, right, fill="─"):
        parts = [fill * metric_width] + [fill * w for w in value_widths]
        return left + mid.join(parts) + right

    def _row(cells):
        parts = [f" {cells[0]:<{metric_width - 1}}"]
        for i, c in enumerate(cells[1:]):
            parts.append(f" {c:>{value_widths[i] - 1}}")
        return "│" + "│".join(parts) + "│"

    total_width = sum([metric_width] + value_widths) + n_val_cols + 1

    lines.append(_border("┌", "┬", "┐"))
    title = f"COMPARACIÓN MULTI-SEED vs BASELINE  (baseline: {n_bl} seed(s), nuevo: {n_nw} seed(s))"
    if len(title) > total_width - 3:
        title = title[: total_width - 6] + "..."
    lines.append(f"│ {title:<{total_width - 3}} │")
    lines.append(f"│ {'NW source: ' + seeds_dir:<{total_width - 3}} │")
    if baseline_dir:
        bl_label = baseline_dir if isinstance(baseline_dir, str) else f"{len(baseline_dir)} rutas"
        bl_text = f"Baseline source: {bl_label}"
        if len(bl_text) > total_width - 4:
            bl_text = bl_text[: total_width - 7] + "..."
        lines.append(f"│ {bl_text:<{total_width - 3}} │")
    lines.append(_border("├", "┼", "┤"))

    lines.append(_row(["Métrica"] + col_headers))
    lines.append(_border("├", "┼", "┤"))

    for (_, name, _), row in zip(key_metrics, table_rows):
        lines.append(_row([name] + row))

    lines.append(_border("└", "┴", "┘"))

    report_text = "\n".join(lines)

    if output_txt:
        os.makedirs(os.path.dirname(output_txt) or ".", exist_ok=True)
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")
    else:
        print(report_text)

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "baseline_entries": baseline_entries,
                    "new_entries": new_entries,
                    "baseline_source": baseline_dir if isinstance(baseline_dir, str) else list(baseline_dir or []),
                },
                f, indent=2,
            )

    return report_text