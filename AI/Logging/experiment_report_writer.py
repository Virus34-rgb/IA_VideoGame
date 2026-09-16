"""
Genera el informe individual (.md) de un experimento -- una plantilla por
EXP_ID en docs/experiments/. Hermano de StatsReportWriter: solo formatea,
no posee estado ni calcula métricas (eso vive en experiment_log_writer.py).

Sección "Config exacta (por seed)":
  A diferencia de la versión anterior (que mostraba el diff de constants.py
  contra los valores por defecto), ahora el informe carga el
  `<seed_dir>/*_config.json` real de cada seed. Como todas las seeds suelen
  compartir la misma config salvo SEED, el render muestra:
    - Una tabla de "valores compartidos" (idénticos en todas las seeds).
    - Una tabla "per-seed" solo con las claves que difieren (típicamente SEED).
"""
import os
from typing import Any, Dict, List, Optional, Tuple


class ExperimentReportWriter:
    @staticmethod
    def write(
        path: str,
        exp_id: str,
        fecha: str,
        objetivo: str,
        hipotesis: str,
        suffix: str,
        seeds: List[str],
        lotes: int,
        n_batch: int,
        baseline_str: str,
        seed_configs: Optional[Dict[str, dict]] = None,
        col_headers: Optional[List[str]] = None,
        table_rows: Optional[List[List[str]]] = None,
        key_metrics: Optional[List[Tuple[str, str, str, str, float]]] = None,
        confusores: str = "",
        conclusion: str = "",
        siguiente_paso: str = "",
        veredicto: str = "N/A",
        veredicto_self: str = "N/A",
        veredicto_rusher: str = "N/A",
        delta_self_pp: Optional[float] = None,
        delta_rusher_pp: Optional[float] = None,
        pre_noisy_fix: bool = False,
        cross_generation: bool = False,
        nota: str = "",
        artefactos_str: str = "",
    ) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        delta_self_str = f"{delta_self_pp:+.2f} pp" if delta_self_pp is not None else "N/A"
        delta_rusher_str = f"{delta_rusher_pp:+.2f} pp" if delta_rusher_pp is not None else "N/A"

        lines: List[str] = [
            f"# {exp_id} — {suffix or '(sin suffix)'}",
            "",
            f"- **Fecha:** {fecha}",
            f"- **Veredicto:** {veredicto}",
            f"- **Veredicto self-play:** {veredicto_self}  (Δ = {delta_self_str})",
            f"- **Veredicto rusher:**   {veredicto_rusher}  (Δ medio 3 bandas = {delta_rusher_str})",
            f"- **Seeds:** {', '.join(seeds) if seeds else 'N/A'}",
            f"- **Lotes por seed:** {lotes}",
            f"- **N (partidas paralelas):** {n_batch}",
            f"- **Baseline:** {baseline_str or '(ninguno, referencia)'}",
        ]

        if pre_noisy_fix:
            lines.append(
                "- **⚠️ Pre-fix NoisyLinear:** este run es de una generación "
                "anterior al fix de cacheo de NoisyLinear (IAV2 o inferior). Los "
                "resultados **no son comparables directamente** con runs IAV3+."
            )
        if cross_generation:
            lines.append(
                "- **⚠️ Comparación cruzada de generaciones:** baseline y run "
                "nuevo son de generaciones IAV distintas. Interpretar con cautela."
            )

        lines += [
            "",
            "## Objetivo / hipótesis",
            "",
            objetivo or "_(sin objetivo -- rellenar EXPERIMENT_OBJETIVO)_",
            "",
            hipotesis or "_(sin hipótesis -- rellenar EXPERIMENT_HIPOTESIS)_",
            "",
            "## Config exacta (por seed)",
            "",
        ]

        lines.extend(ExperimentReportWriter._render_seed_configs(seed_configs or {}, seeds))

        lines += ["", "## Resultado", ""]

        if table_rows and col_headers and key_metrics:
            lines.append("| Métrica | " + " | ".join(col_headers) + " |")
            lines.append("|---" * (len(col_headers) + 1) + "|")
            # key_metrics ahora es lista de 5-tuplas
            # (section, key, display_name, fmt, scale) -- usamos display_name
            # (índice 2) como nombre de fila visible.
            for metric_tuple, row_vals in zip(key_metrics, table_rows):
                display_name = metric_tuple[2]
                lines.append(f"| {display_name} | " + " | ".join(row_vals) + " |")
        else:
            lines.append("_Sin datos de resultado (¿baseline y seeds nuevas vacíos?)._")

        lines += [
            "",
            "## Confusores conocidos",
            "",
            confusores or "_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_",
            "",
            "## Conclusión y siguiente paso",
            "",
            conclusion or "_(pendiente -- rellenar EXPERIMENT_CONCLUSION)_",
            "",
            siguiente_paso or "_(pendiente -- rellenar EXPERIMENT_SIGUIENTE_PASO)_",
            "",
            "## Nota",
            "",
            nota or "_(sin notas)_",
            "",
            "## Artefactos (carpetas por seed)",
            "",
        ]
        if artefactos_str:
            for art in artefactos_str.split(";"):
                lines.append(f"- `{art}`")
        else:
            lines.append("_(sin artefactos registrados)_")
        lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ------------------------------------------------------------
    # Render de la sección "Config exacta (por seed)"
    # ------------------------------------------------------------
    @staticmethod
    def _render_seed_configs(seed_configs: Dict[str, dict], seeds: List[str]) -> List[str]:
        """
        Renderiza la sección de config por seed a partir del dict cargado en
        experiment_log_writer._load_seed_configs().

        Reglas:
          - Una clave cuyo valor es idéntico en TODAS las seeds va a la tabla
            "valores compartidos".
          - Una clave cuyo valor difiere (típicamente SEED) va a "valores
            específicos por seed".

        Acepta labels con o sin prefijo 's' (normaliza a 's<seed>').
        """
        if not seed_configs:
            return [
                "_No se encontraron configs por seed en "
                "`<base_path>/s<seed>/*_config.json`._"
            ]

        # Normalizar labels al formato de las claves del dict ("s42", "s43"...).
        seeds_normalized: List[str] = []
        for s in seeds:
            label = str(s) if str(s).startswith("s") else f"s{s}"
            if label in seed_configs:
                seeds_normalized.append(label)
        if not seeds_normalized:
            return [
                "_Los configs disponibles no coinciden con las seeds del informe. "
                f"Disponibles: {sorted(seed_configs.keys())}._"
            ]

        # Unión de todas las claves de todos los configs.
        all_keys = set()
        for cfg in seed_configs.values():
            all_keys |= set(cfg.keys())
        all_keys_sorted = sorted(all_keys)

        shared: Dict[str, Any] = {}
        per_seed: Dict[str, Dict[str, Any]] = {}

        for k in all_keys_sorted:
            values = [seed_configs[s].get(k, "<missing>") for s in seeds_normalized]
            first = values[0]
            if all(v == first for v in values):
                shared[k] = first
            else:
                per_seed[k] = dict(zip(seeds_normalized, values))

        out: List[str] = []
        out.append(
            f"Config cargada de `<base_path>/s<seed>/*_config.json` de cada seed. "
            f"Seeds incluidas: {', '.join(seeds_normalized)}."
        )
        out.append("")

        if shared:
            out.append("### Valores compartidos (idénticos en todas las seeds)")
            out.append("")
            out.append("| Constante | Valor |")
            out.append("|---|---|")
            for k in sorted(shared.keys()):
                out.append(f"| `{k}` | `{shared[k]}` |")

        if per_seed:
            out.append("")
            out.append("### Valores específicos por seed")
            out.append("")
            out.append("| Constante | " + " | ".join(seeds_normalized) + " |")
            out.append("|---" * (len(seeds_normalized) + 1) + "|")
            for k in sorted(per_seed.keys()):
                vals = per_seed[k]
                cells = " | ".join(f"`{vals.get(s, '')}`" for s in seeds_normalized)
                out.append(f"| `{k}` | {cells} |")

        return out