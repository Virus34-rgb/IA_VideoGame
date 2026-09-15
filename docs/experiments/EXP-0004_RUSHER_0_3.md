# EXP-0004 — RUSHER_0.3

- **Fecha:** 2026-09-13
- **Veredicto:** negativo
- **Seeds:** 42, 43, 44
- **Lotes por seed:** 100
- **N (partidas paralelas):** 2048
- **Baseline:** models\IAV2_BASE_100

## Objetivo / hipótesis

BASE pero con enemigos rusher como posibilidades, quiero estudiar si el cambio en la memoria mejora los resultados

El cambio en la memoria de turnos mejora el aprendizaje con rusher

## Config exacta (diff vs. constants.py por defecto)

| Constante | Valor por defecto | Valor en este experimento |
|---|---|---|
| `DELETE_DIRECTORIES` | `True` | `False` |
| `EVAL_EPISODES` | `2` | `20` |
| `PROFILE_CPROFILE` | `True` | `False` |
| `PROFILE_CPROFILE_OUTPUT` | `profile_cpu.prof` | `optimizedAll2.prof` |
| `RUSHER_FINETUNE_EPISODES` | `300` | `5` |
| `RUSHER_TEST_EPISODES` | `100` | `20` |
| `RUSHER_TEST_EPSILON` | `0.05` | `0.0` |
| `TRAIN_EPISODES` | `20` | `100` |
| `USE_WANDB` | `True` | `False` |

## Resultado

| Métrica | BL_s42 | BL_s43 | BL_s44 | BL_μ | NW_s42 | NW_s43 | NW_s44 | NW_μ | Δμ |
|---|---|---|---|---|---|---|---|---|---|
| Win ratio P1 (sin empates) | 68.25 | 78.66 | 48.20 | 65.04 | 28.19 | 80.72 | 57.15 | 55.35 | -9.68 |
| Turnos medios por partida | 9.18 | 8.60 | 7.59 | 8.46 | 8.13 | 9.36 | 9.19 | 8.89 | 0.44 |
| Elo P1 | 979.4 | 1018.2 | 958.1 | 985.2 | 951.9 | 1013.4 | 960.7 | 975.3 | -9.9 |
| Elo P2 | 940.4 | 934.3 | 943.2 | 939.3 | 991.4 | 944.7 | 938.5 | 958.2 | 18.9 |
| Daño medio P1 | 104.96 | 104.76 | 96.44 | 102.05 | 75.35 | 103.36 | 99.93 | 92.88 | -9.17 |
| Reward media P1 | 2.98 | 5.50 | -0.92 | 2.52 | -5.89 | 5.67 | 0.51 | 0.10 | -2.42 |
| Bajas medias P1 | 1.49 | 1.06 | 1.91 | 1.49 | 2.39 | 1.04 | 1.78 | 1.74 | 0.25 |
| Daño por sobrekill medio P1 | 7.34 | 8.05 | 5.94 | 7.11 | 3.67 | 8.35 | 6.42 | 6.15 | -0.96 |
| Kill confirmed medio P1 | 2.24 | 2.49 | 1.83 | 2.19 | 1.13 | 2.53 | 1.96 | 1.87 | -0.31 |
| Defensas desperdiciadas P1 | 1.56 | 1.67 | 1.13 | 1.45 | 0.94 | 1.88 | 1.43 | 1.42 | -0.04 |
| Movimientos estratégicos P1 (%) | 9.11 | 11.74 | 12.27 | 11.04 | 7.50 | 10.80 | 11.30 | 9.87 | -1.17 |
| Victorias IA (2nd) | 0.6879 | 0.5901 | 0.5472 | 0.6084 | 0.4647 | 0.6423 | 0.5806 | 0.5626 | -0.0458 |
| Victorias IA (2nd) | 0.1890 | 0.1515 | 0.1395 | 0.1600 | 0.0079 | 0.1908 | 0.1123 | 0.1037 | -0.0563 |
| Victorias IA (2nd) | 0.1146 | 0.0898 | 0.1008 | 0.1017 | 0.0017 | 0.1174 | 0.0790 | 0.0660 | -0.0357 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

Se observa que la red solo colapsa con finetune en la seed 42, en las otras dos seeds disminuye pero no en tanta medida

Revisar con fineturne integrado tras probar las recompensas interpoladas a la agresividad

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_RUSHER_0_3\s42`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_RUSHER_0_3\s43`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_RUSHER_0_3\s44`
