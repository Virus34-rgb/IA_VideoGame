# Golden files — Tests de regresión

Este directorio contiene **salidas de referencia** ("golden") que los tests
de regresión comparan contra el output actual del motor.

## Ficheros

- `stats_rusher_aggr_05_seed42.txt` — `stats2.txt` producido por un lote
  determinista de N=64 contra `PlayerRusherV(aggression=0.5)` con seed 42.
  Test asociado: `tests/test_regression_rusher.py`.

Este fichero **se genera automáticamente**, no se edita a mano.

## Cómo regenerar

Solo tras un **cambio intencional** en el motor:

```bash
python tests/regenerate_golden.py
git diff tests/golden/
```

# Escenario A — "Voy a tocar el motor y quiero saber si lo rompo"

## 1. Antes de empezar, sanity check: ¿el test pasa con el código actual?
```bash
python -m pytest tests/test_regression_rusher.py -v
```
# → PASSED

# 2. Editas resolve_actions.py, vectorizedEnvironment.py, etc.

# 3. Después del cambio, corre el test otra vez
```bash
python -m pytest tests/test_regression_rusher.py -v
```

# 4a. Si sigue PASSED → tu cambio no altera el motor. Puedes seguir.
# 4b. Si FAILED → lee el diff que imprime el test.

## Escenario B — "El test FAILED, y el cambio es intencional"

# 1. Mira el diff del test. Verás algo tipo:
#    Daño medio P1: golden=42.31, actual=48.72, delta=+6.41 (tol=0.01)

# 2. ¿El cambio es lo que esperabas? Sí. Regenera el golden.
```bash
python tests/regenerate_golden.py
```
# 3. Revisa que el golden cambió de verdad
```bash
git diff tests/golden/stats_rusher_aggr_05_seed42.txt
```
# 4. Commitea código + golden JUNTOS en el mismo commit
```bash
git add AI/Environment/resolve_actions.py tests/golden/stats_rusher_aggr_05_seed42.txt
git commit -m "fix: rango de Fireball +6dmg → actualiza golden"
```

## Escenario C — "El test FAILED, y NO entiendo por qué"

# 1. NO regeneres nada
# 2. Mira las métricas que cambiaron
```bash
python -m pytest tests/test_regression_rusher.py -v -s
```