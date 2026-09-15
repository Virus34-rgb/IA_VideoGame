# EXP-0017 — Eval_BETA_START_0.8

- **Fecha:** 2026-09-15
- **Veredicto:** negativo
- **Seeds:** 42, 43, 44
- **Lotes por seed:** 200
- **N (partidas paralelas):** 2048
- **Baseline:** models\IAV2_BASE_200

## Objetivo / hipótesis

BASE con BETA_START = 0.8 

La varianza del PER inicial distorisona el aprendizaje 

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
| `TRAIN_EPISODES` | `20` | `200` |
| `USE_WANDB` | `True` | `False` |

## Resultado

| Métrica | BL_s42 | BL_s43 | BL_s44 | BL_μ | NW_s42 | NW_s43 | NW_s44 | NW_μ | Δμ |
|---|---|---|---|---|---|---|---|---|---|
| Win ratio P1 (sin empates) | 56.82 | 84.17 | 49.84 | 63.61 | 38.65 | 77.54 | 54.61 | 56.93 | -6.68 |
| Turnos medios por partida | 8.68 | 9.11 | 8.50 | 8.76 | 8.23 | 8.94 | 8.21 | 8.46 | -0.30 |
| Elo P1 | 933.9 | 1001.6 | 923.8 | 953.1 | 938.4 | 1057.1 | 912.6 | 969.4 | 16.3 |
| Elo P2 | 934.8 | 914.1 | 912.4 | 920.4 | 929.7 | 954.8 | 892.4 | 925.6 | 5.2 |
| Daño medio P1 | 99.37 | 108.57 | 95.99 | 101.31 | 81.26 | 106.19 | 96.50 | 94.65 | -6.66 |
| Reward media P1 | 0.64 | 6.52 | -0.94 | 2.07 | -3.57 | 5.17 | 0.21 | 0.60 | -1.47 |
| Bajas medias P1 | 1.75 | 1.07 | 1.94 | 1.59 | 2.12 | 1.10 | 1.84 | 1.69 | 0.10 |
| Daño por sobrekill medio P1 | 6.61 | 8.19 | 6.15 | 6.98 | 4.50 | 7.94 | 6.43 | 6.29 | -0.69 |
| Kill confirmed medio P1 | 1.99 | 2.51 | 1.89 | 2.13 | 1.37 | 2.45 | 1.95 | 1.92 | -0.21 |
| Defensas desperdiciadas P1 | 1.41 | 1.79 | 1.21 | 1.47 | 1.03 | 1.73 | 1.34 | 1.37 | -0.10 |
| Movimientos estratégicos P1 (%) | 7.91 | 11.28 | 11.98 | 10.39 | 9.04 | 11.54 | 13.10 | 11.23 | 0.84 |
| Victorias IA (2nd) | 0.6392 | 0.6789 | 0.6235 | 0.6472 | 0.4656 | 0.5874 | 0.5613 | 0.5381 | -0.1091 |
| Victorias IA (2nd) | 0.1611 | 0.1625 | 0.1914 | 0.1717 | 0.0763 | 0.1692 | 0.1972 | 0.1476 | -0.0241 |
| Victorias IA (2nd) | 0.0896 | 0.0679 | 0.1242 | 0.0939 | 0.0681 | 0.0873 | 0.1298 | 0.0951 | 0.0011 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

Resultados negativos tambien, descartado por ahora

Por decidir

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_Eval_BETA_START_0_8\s42`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_Eval_BETA_START_0_8\s43`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_Eval_BETA_START_0_8\s44`
