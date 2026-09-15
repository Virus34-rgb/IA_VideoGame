# EXP-0009 — CDQNS_120

- **Fecha:** 2026-09-13
- **Veredicto:** neutro
- **Seeds:** 42, 43, 44
- **Lotes por seed:** 100
- **N (partidas paralelas):** 2048
- **Baseline:** models\IAV2_BASE_100

## Objetivo / hipótesis

Base con CDQN Sel 120

CDQN Sel 120 mejor a el resultado combiando con la memoria

## Config exacta (diff vs. constants.py por defecto)

| Constante | Valor por defecto | Valor en este experimento |
|---|---|---|
| `DELETE_DIRECTORIES` | `True` | `False` |
| `EVAL_EPISODES` | `2` | `20` |
| `PROFILE_CPROFILE` | `True` | `False` |
| `PROFILE_CPROFILE_OUTPUT` | `profile_cpu.prof` | `optimizedAll2.prof` |
| `RUSHER_FINETUNE_EPISODES` | `300` | `5` |
| `RUSHER_OPPONENT_PERCENTAGE` | `0.3` | `0.0` |
| `RUSHER_TEST_EPISODES` | `100` | `20` |
| `RUSHER_TEST_EPSILON` | `0.05` | `0.0` |
| `TRAIN_EPISODES` | `20` | `100` |
| `USE_WANDB` | `True` | `False` |

## Resultado

| Métrica | BL_s42 | BL_s43 | BL_s44 | BL_μ | NW_s42 | NW_s43 | NW_s44 | NW_μ | Δμ |
|---|---|---|---|---|---|---|---|---|---|
| Win ratio P1 (sin empates) | 68.25 | 78.66 | 48.20 | 65.04 | 66.87 | 80.45 | 51.76 | 66.36 | 1.32 |
| Turnos medios por partida | 9.18 | 8.60 | 7.59 | 8.46 | 8.17 | 9.52 | 7.43 | 8.37 | -0.08 |
| Elo P1 | 979.4 | 1018.2 | 958.1 | 985.2 | 954.3 | 1048.8 | 937.2 | 980.1 | -5.1 |
| Elo P2 | 940.4 | 934.3 | 943.2 | 939.3 | 953.0 | 934.4 | 932.3 | 939.9 | 0.6 |
| Daño medio P1 | 104.96 | 104.76 | 96.44 | 102.05 | 98.90 | 107.77 | 93.28 | 99.98 | -2.07 |
| Reward media P1 | 2.98 | 5.50 | -0.92 | 2.52 | 2.92 | 5.63 | -0.22 | 2.78 | 0.26 |
| Bajas medias P1 | 1.49 | 1.06 | 1.91 | 1.49 | 1.42 | 1.05 | 1.82 | 1.43 | -0.06 |
| Daño por sobrekill medio P1 | 7.34 | 8.05 | 5.94 | 7.11 | 6.81 | 8.13 | 5.98 | 6.97 | -0.14 |
| Kill confirmed medio P1 | 2.24 | 2.49 | 1.83 | 2.19 | 2.06 | 2.52 | 1.83 | 2.14 | -0.05 |
| Defensas desperdiciadas P1 | 1.56 | 1.67 | 1.13 | 1.45 | 1.43 | 1.86 | 1.15 | 1.48 | 0.03 |
| Movimientos estratégicos P1 (%) | 9.11 | 11.74 | 12.27 | 11.04 | 9.08 | 11.17 | 12.20 | 10.82 | -0.22 |
| Victorias IA (2nd) | 0.6879 | 0.5901 | 0.5472 | 0.6084 | 0.5844 | 0.6673 | 0.5666 | 0.6061 | -0.0023 |
| Victorias IA (2nd) | 0.1890 | 0.1515 | 0.1395 | 0.1600 | 0.0895 | 0.1734 | 0.1335 | 0.1321 | -0.0278 |
| Victorias IA (2nd) | 0.1146 | 0.0898 | 0.1008 | 0.1017 | 0.0512 | 0.1003 | 0.0900 | 0.0805 | -0.0212 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

_(pendiente -- rellenar EXPERIMENT_CONCLUSION)_

Evaluacion de CDQN_SEL 240

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNS_120\s42`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNS_120\s43`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNS_120\s44`
