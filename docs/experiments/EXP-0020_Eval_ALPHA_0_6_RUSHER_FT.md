# EXP-0020 — Eval_ALPHA_0.6_RUSHER_FT

- **Fecha:** 2026-09-15
- **Veredicto:** negativo
- **Seeds:** 42, 43, 44
- **Lotes por seed:** 200
- **N (partidas paralelas):** 2048
- **Baseline:** models\IAV2_BASE_200

## Objetivo / hipótesis

BASE con ALPHA = 0.6 FINESTEP

La variable de prioridad tan alta perjudica el aprendizaje contra rusher 

## Config exacta (diff vs. constants.py por defecto)

| Constante | Valor por defecto | Valor en este experimento |
|---|---|---|
| `DELETE_DIRECTORIES` | `True` | `False` |
| `EVAL_EPISODES` | `2` | `20` |
| `PROFILE_CPROFILE` | `True` | `False` |
| `PROFILE_CPROFILE_OUTPUT` | `profile_cpu.prof` | `optimizedAll2.prof` |
| `RUN_RUSHER_FINETUNE` | `False` | `True` |
| `RUSHER_FINETUNE_EPISODES` | `300` | `100` |
| `RUSHER_OPPONENT_PERCENTAGE` | `0.3` | `0.0` |
| `RUSHER_TEST_EPISODES` | `100` | `20` |
| `RUSHER_TEST_EPSILON` | `0.05` | `0.0` |
| `TRAIN_EPISODES` | `20` | `200` |
| `USE_WANDB` | `True` | `False` |

## Resultado

| Métrica | BL_s42 | BL_s43 | BL_s44 | BL_μ | NW_s42 | NW_s43 | NW_s44 | NW_μ | Δμ |
|---|---|---|---|---|---|---|---|---|---|
| Win ratio P1 (sin empates) | 56.82 | 84.17 | 49.84 | 63.61 | 4.70 | 91.17 | 17.34 | 37.74 | -25.87 |
| Turnos medios por partida | 8.68 | 9.11 | 8.50 | 8.76 | 7.70 | 8.61 | 8.20 | 8.17 | -0.59 |
| Elo P1 | 933.9 | 1001.6 | 923.8 | 953.1 | 927.6 | 1057.3 | 931.9 | 972.3 | 19.2 |
| Elo P2 | 934.8 | 914.1 | 912.4 | 920.4 | 937.9 | 913.9 | 896.0 | 915.9 | -4.5 |
| Daño medio P1 | 99.37 | 108.57 | 95.99 | 101.31 | 55.44 | 107.17 | 77.65 | 80.09 | -21.22 |
| Reward media P1 | 0.64 | 6.52 | -0.94 | 2.07 | -11.16 | 8.29 | -7.95 | -3.61 | -5.68 |
| Bajas medias P1 | 1.75 | 1.07 | 1.94 | 1.59 | 2.91 | 0.66 | 2.70 | 2.09 | 0.50 |
| Daño por sobrekill medio P1 | 6.61 | 8.19 | 6.15 | 6.98 | 1.26 | 9.26 | 3.31 | 4.61 | -2.37 |
| Kill confirmed medio P1 | 1.99 | 2.51 | 1.89 | 2.13 | 0.41 | 2.74 | 1.03 | 1.39 | -0.74 |
| Defensas desperdiciadas P1 | 1.41 | 1.79 | 1.21 | 1.47 | 0.51 | 2.23 | 0.87 | 1.20 | -0.27 |
| Movimientos estratégicos P1 (%) | 7.91 | 11.28 | 11.98 | 10.39 | 9.17 | 11.82 | 12.48 | 11.16 | 0.77 |
| Victorias IA (2nd) | 0.6392 | 0.6789 | 0.6235 | 0.6472 | 0.0815 | 0.6465 | 0.2723 | 0.3334 | -0.3138 |
| Victorias IA (2nd) | 0.1611 | 0.1625 | 0.1914 | 0.1717 | 0.0031 | 0.2023 | 0.0133 | 0.0729 | -0.0987 |
| Victorias IA (2nd) | 0.0896 | 0.0679 | 0.1242 | 0.0939 | 0.0011 | 0.1149 | 0.0097 | 0.0419 | -0.0520 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

Descartado

Evaluar con ALPHA = 0.4 o evaluar con rusher 0.3 y rusher train steps

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_Eval_ALPHA_0_6_RUSHER_FT\s42`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_Eval_ALPHA_0_6_RUSHER_FT\s43`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_Eval_ALPHA_0_6_RUSHER_FT\s44`
