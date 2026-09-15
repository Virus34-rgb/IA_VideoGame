# EXP-0016 — Eval_BETA_START_0.6

- **Fecha:** 2026-09-15
- **Veredicto:** neutro
- **Seeds:** 42, 43, 44
- **Lotes por seed:** 200
- **N (partidas paralelas):** 2048
- **Baseline:** models\IAV2_BASE_200

## Objetivo / hipótesis

BASE con BETA_START = 0.6 

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
| Win ratio P1 (sin empates) | 56.82 | 84.17 | 49.84 | 63.61 | 54.37 | 84.33 | 56.72 | 65.14 | 1.53 |
| Turnos medios por partida | 8.68 | 9.11 | 8.50 | 8.76 | 8.20 | 9.21 | 7.58 | 8.33 | -0.43 |
| Elo P1 | 933.9 | 1001.6 | 923.8 | 953.1 | 920.2 | 990.0 | 944.0 | 951.4 | -1.7 |
| Elo P2 | 934.8 | 914.1 | 912.4 | 920.4 | 941.7 | 917.0 | 903.6 | 920.8 | 0.3 |
| Daño medio P1 | 99.37 | 108.57 | 95.99 | 101.31 | 90.70 | 106.94 | 94.21 | 97.28 | -4.03 |
| Reward media P1 | 0.64 | 6.52 | -0.94 | 2.07 | 0.01 | 6.54 | 0.78 | 2.44 | 0.37 |
| Bajas medias P1 | 1.75 | 1.07 | 1.94 | 1.59 | 1.75 | 0.94 | 1.71 | 1.47 | -0.12 |
| Daño por sobrekill medio P1 | 6.61 | 8.19 | 6.15 | 6.98 | 5.75 | 8.34 | 6.49 | 6.86 | -0.12 |
| Kill confirmed medio P1 | 1.99 | 2.51 | 1.89 | 2.13 | 1.74 | 2.55 | 1.96 | 2.08 | -0.05 |
| Defensas desperdiciadas P1 | 1.41 | 1.79 | 1.21 | 1.47 | 1.26 | 1.89 | 1.25 | 1.47 | -0.00 |
| Movimientos estratégicos P1 (%) | 7.91 | 11.28 | 11.98 | 10.39 | 7.91 | 11.20 | 11.26 | 10.12 | -0.27 |
| Victorias IA (2nd) | 0.6392 | 0.6789 | 0.6235 | 0.6472 | 0.5947 | 0.6977 | 0.6045 | 0.6323 | -0.0149 |
| Victorias IA (2nd) | 0.1611 | 0.1625 | 0.1914 | 0.1717 | 0.0627 | 0.1953 | 0.1787 | 0.1456 | -0.0261 |
| Victorias IA (2nd) | 0.0896 | 0.0679 | 0.1242 | 0.0939 | 0.0662 | 0.1039 | 0.1286 | 0.0996 | 0.0056 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

Los resultados son peores que con BetaStart 1.0, probare con 0.8.

Por decidir

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_Eval_BETA_START_0_6\s42`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_Eval_BETA_START_0_6\s43`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_Eval_BETA_START_0_6\s44`
