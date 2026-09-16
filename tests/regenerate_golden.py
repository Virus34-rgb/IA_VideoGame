"""
Regenera el golden del test de regresión dorado contra el rusher 0.5.

Uso:
    python tests/regenerate_golden.py

Reusa EXACTAMENTE el mismo setup que test_regression_rusher.py (importa su
_run_deterministic_eval), así que el golden generado y el test siempre
parten del mismo estado. Sobrescribe el golden con el resultado actual.

Cuándo correrlo:
  Solo tras un cambio INTENCIONAL en el motor del juego (fórmulas, masking,
  encoding, etc.), y SOLO después de haber revisado el diff métrica a
  métrica para confirmar que el cambio es el que querías.

Cuándo NO correrlo:
  Cuando el test falla y no entiendes por qué. En ese caso, el test está
  haciendo su trabajo — hay un bug.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Permitir `python tests/regenerate_golden.py` sin instalar el proyecto.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_regression_rusher import (  # noqa: E402
    GOLDEN_PATH,
    _run_deterministic_eval,
)


def main() -> int:
    print(f"[regenerate_golden] Ejecutando eval determinista...")
    print(f"[regenerate_golden] Golden destino: {GOLDEN_PATH}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        stats_path = _run_deterministic_eval(tmp_path)
        if not stats_path.exists():
            print(f"[ERROR] No se generó {stats_path}", file=sys.stderr)
            return 1
        content = stats_path.read_text(encoding="utf-8")

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)

    if GOLDEN_PATH.exists():
        old = GOLDEN_PATH.read_text(encoding="utf-8")
        if old == content:
            print("[regenerate_golden] El golden no ha cambiado (byte-idéntico).")
            return 0
        print("[regenerate_golden] El golden CAMBIA. Revisa el diff abajo.")

    GOLDEN_PATH.write_text(content, encoding="utf-8")
    print(f"[regenerate_golden] Golden escrito ({len(content)} bytes).")
    print(f"[regenerate_golden] Revisa con: git diff {GOLDEN_PATH}")
    print(
        "[regenerate_golden] Si el diff refleja tu cambio intencional, "
        "commitea golden + código en el mismo commit."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())