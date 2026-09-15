# EXP-0007 — CDQNT_150

- **Fecha:** 2026-09-13
- **Veredicto:** positivo
- **Seeds:** 42, 43, 44
- **Lotes por seed:** 100
- **N (partidas paralelas):** 2048
- **Baseline:** models\IAV2_BASE_100

## Objetivo / hipótesis

Base con CDQN Turn 150

CDQN Turn 150 mejor ael resultado combiando con la memoria

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
| Win ratio P1 (sin empates) | 68.25 | 78.66 | 48.20 | 65.04 | 63.17 | 84.79 | 63.57 | 70.51 | 5.47 |
| Turnos medios por partida | 9.18 | 8.60 | 7.59 | 8.46 | 7.92 | 9.36 | 8.09 | 8.46 | 0.00 |
| Elo P1 | 979.4 | 1018.2 | 958.1 | 985.2 | 966.5 | 1043.4 | 986.6 | 998.8 | 13.6 |
| Elo P2 | 940.4 | 934.3 | 943.2 | 939.3 | 952.0 | 934.1 | 931.6 | 939.2 | -0.1 |
| Daño medio P1 | 104.96 | 104.76 | 96.44 | 102.05 | 96.66 | 108.01 | 102.56 | 102.41 | 0.36 |
| Reward media P1 | 2.98 | 5.50 | -0.92 | 2.52 | 2.18 | 6.62 | 2.26 | 3.69 | 1.17 |
| Bajas medias P1 | 1.49 | 1.06 | 1.91 | 1.49 | 1.46 | 0.92 | 1.54 | 1.31 | -0.18 |
| Daño por sobrekill medio P1 | 7.34 | 8.05 | 5.94 | 7.11 | 6.57 | 8.46 | 7.01 | 7.35 | 0.24 |
| Kill confirmed medio P1 | 2.24 | 2.49 | 1.83 | 2.19 | 1.99 | 2.59 | 2.15 | 2.24 | 0.06 |
| Defensas desperdiciadas P1 | 1.56 | 1.67 | 1.13 | 1.45 | 1.35 | 1.98 | 1.40 | 1.58 | 0.12 |
| Movimientos estratégicos P1 (%) | 9.11 | 11.74 | 12.27 | 11.04 | 8.65 | 10.61 | 12.97 | 10.74 | -0.30 |
| Victorias IA (2nd) | 0.6879 | 0.5901 | 0.5472 | 0.6084 | 0.5683 | 0.6820 | 0.6485 | 0.6329 | 0.0246 |
| Victorias IA (2nd) | 0.1890 | 0.1515 | 0.1395 | 0.1600 | 0.1388 | 0.1911 | 0.1799 | 0.1699 | 0.0100 |
| Victorias IA (2nd) | 0.1146 | 0.0898 | 0.1008 | 0.1017 | 0.0918 | 0.1094 | 0.1177 | 0.1063 | 0.0046 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

_(pendiente -- rellenar EXPERIMENT_CONCLUSION)_

CDQN Turn 300

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNT_150\s42`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNT_150\s43`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNT_150\s44`
