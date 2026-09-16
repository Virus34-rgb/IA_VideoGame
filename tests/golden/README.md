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