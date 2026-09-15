# EXP-0018 — Eval_ALPHA_0.6

- **Fecha:** 2026-09-15
- **Veredicto:** neutro
- **Seeds:** 42, 43, 44
- **Lotes por seed:** 200
- **N (partidas paralelas):** 2048
- **Baseline:** models\IAV2_BASE_200

## Objetivo / hipótesis

BASE con ALPHA = 0.6 

La variable de prioridad tan alta perjudica el aprendizaje contra rusher 

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
| Win ratio P1 (sin empates) | 56.82 | 84.17 | 49.84 | 63.61 | 39.92 | 89.47 | 56.09 | 61.83 | -1.78 |
| Turnos medios por partida | 8.68 | 9.11 | 8.50 | 8.76 | 8.77 | 9.07 | 7.79 | 8.54 | -0.22 |
| Elo P1 | 933.9 | 1001.6 | 923.8 | 953.1 | 927.6 | 1057.3 | 931.9 | 972.3 | 19.2 |
| Elo P2 | 934.8 | 914.1 | 912.4 | 920.4 | 937.9 | 913.9 | 896.0 | 915.9 | -4.5 |
| Daño medio P1 | 99.37 | 108.57 | 95.99 | 101.31 | 88.92 | 106.85 | 97.21 | 97.66 | -3.65 |
| Reward media P1 | 0.64 | 6.52 | -0.94 | 2.07 | -3.26 | 7.74 | 0.73 | 1.74 | -0.34 |
| Bajas medias P1 | 1.75 | 1.07 | 1.94 | 1.59 | 2.12 | 0.78 | 1.72 | 1.54 | -0.05 |
| Daño por sobrekill medio P1 | 6.61 | 8.19 | 6.15 | 6.98 | 5.09 | 8.80 | 6.53 | 6.81 | -0.18 |
| Kill confirmed medio P1 | 1.99 | 2.51 | 1.89 | 2.13 | 1.54 | 2.72 | 1.97 | 2.08 | -0.05 |
| Defensas desperdiciadas P1 | 1.41 | 1.79 | 1.21 | 1.47 | 1.22 | 2.13 | 1.26 | 1.54 | 0.07 |
| Movimientos estratégicos P1 (%) | 7.91 | 11.28 | 11.98 | 10.39 | 8.35 | 11.38 | 12.74 | 10.82 | 0.43 |
| Victorias IA (2nd) | 0.6392 | 0.6789 | 0.6235 | 0.6472 | 0.5498 | 0.6250 | 0.4365 | 0.5371 | -0.1101 |
| Victorias IA (2nd) | 0.1611 | 0.1625 | 0.1914 | 0.1717 | 0.0724 | 0.1349 | 0.1478 | 0.1184 | -0.0533 |
| Victorias IA (2nd) | 0.0896 | 0.0679 | 0.1242 | 0.0939 | 0.0323 | 0.0664 | 0.1202 | 0.0730 | -0.0210 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

No produce recompensas positivas sin entrenamiento contra rusher.

Evaluar con ALPHA = 0.4 o evaluar con rusher 0.3 y rusher train steps

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_Eval_ALPHA_0_6\s42`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_Eval_ALPHA_0_6\s43`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_Eval_ALPHA_0_6\s44`
