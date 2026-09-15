"""
Genera el informe individual (.md) de un experimento -- una plantilla por
EXP_ID en docs/experiments/. Hermano de StatsReportWriter: solo formatea,
no posee estado ni calcula métricas (eso vive en experiment_log_writer.py).
"""
import os
from typing import Any, Dict, List, Tuple


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
        constants_diff: Dict[str, Tuple[Any, Any]],
        col_headers: List[str],
        table_rows: List[List[str]],
        key_metrics: List[Tuple[str, str, str]],
        confusores: str,
        conclusion: str,
        siguiente_paso: str,
        veredicto: str,
        nota: str,
        artefactos_str: str,
    ) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        lines = [
            f"# {exp_id} — {suffix or '(sin suffix)'}",
            "",
            f"- **Fecha:** {fecha}",
            f"- **Veredicto:** {veredicto}",
            f"- **Seeds:** {', '.join(seeds) if seeds else 'N/A'}",
            f"- **Lotes por seed:** {lotes}",
            f"- **N (partidas paralelas):** {n_batch}",
            f"- **Baseline:** {baseline_str or '(ninguno, referencia)'}",
            "",
            "## Objetivo / hipótesis",
            "",
            objetivo or "_(sin objetivo -- rellenar EXPERIMENT_OBJETIVO)_",
            "",
            hipotesis or "_(sin hipótesis -- rellenar EXPERIMENT_HIPOTESIS)_",
            "",
            "## Config exacta (diff vs. constants.py por defecto)",
            "",
        ]

        # constants_diff viene de AI.Logging.constants_diff.compute_diff():
        # solo contiene claves que REALMENTE difieren del constants.py base,
        # así que un dict vacío aquí es información real ("sin overrides"),
        # no un fallo de recolección.
        if constants_diff:
            lines.append("| Constante | Valor por defecto | Valor en este experimento |")
            lines.append("|---|---|---|")
            for name in sorted(constants_diff.keys()):
                default_v, current_v = constants_diff[name]
                lines.append(f"| `{name}` | `{default_v}` | `{current_v}` |")
        else:
            lines.append("_Sin diferencias respecto a los valores por defecto de constants.py._")

        lines += ["", "## Resultado", ""]

        # key_metrics y table_rows vienen ya alineados posicionalmente
        # (misma función build_metrics_table que usa el comparador de
        # consola) -- zip es seguro porque ambas listas se generan juntas.
        if table_rows:
            lines.append("| Métrica | " + " | ".join(col_headers) + " |")
            lines.append("|---" * (len(col_headers) + 1) + "|")
            for (_, name, _), row_vals in zip(key_metrics, table_rows):
                lines.append(f"| {name} | " + " | ".join(row_vals) + " |")
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