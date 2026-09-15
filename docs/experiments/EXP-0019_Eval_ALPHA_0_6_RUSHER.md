# EXP-0019 — Eval_ALPHA_0.6_RUSHER

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
| `RUSHER_TEST_EPISODES` | `100` | `20` |
| `RUSHER_TEST_EPSILON` | `0.05` | `0.0` |
| `TRAIN_EPISODES` | `20` | `200` |
| `USE_WANDB` | `True` | `False` |

## Resultado

| Métrica | BL_s42 | BL_s43 | BL_s44 | BL_μ | NW_s42 | NW_s43 | NW_s44 | NW_μ | Δμ |
|---|---|---|---|---|---|---|---|---|---|
| Win ratio P1 (sin empates) | 56.82 | 84.17 | 49.84 | 63.61 | 59.55 | 84.67 | 49.71 | 64.64 | 1.03 |
| Turnos medios por partida | 8.68 | 9.11 | 8.50 | 8.76 | 9.44 | 9.15 | 8.64 | 9.08 | 0.31 |
| Elo P1 | 933.9 | 1001.6 | 923.8 | 953.1 | 937.6 | 1024.4 | 937.8 | 966.6 | 13.5 |
| Elo P2 | 934.8 | 914.1 | 912.4 | 920.4 | 924.6 | 911.2 | 899.2 | 911.7 | -8.8 |
| Daño medio P1 | 99.37 | 108.57 | 95.99 | 101.31 | 99.60 | 107.71 | 97.42 | 101.58 | 0.27 |
| Reward media P1 | 0.64 | 6.52 | -0.94 | 2.07 | 0.91 | 6.67 | -0.90 | 2.23 | 0.15 |
| Bajas medias P1 | 1.75 | 1.07 | 1.94 | 1.59 | 1.73 | 0.91 | 1.92 | 1.52 | -0.07 |
| Daño por sobrekill medio P1 | 6.61 | 8.19 | 6.15 | 6.98 | 6.70 | 8.53 | 6.14 | 7.12 | 0.14 |
| Kill confirmed medio P1 | 1.99 | 2.51 | 1.89 | 2.13 | 2.02 | 2.60 | 1.87 | 2.16 | 0.03 |
| Defensas desperdiciadas P1 | 1.41 | 1.79 | 1.21 | 1.47 | 1.61 | 1.88 | 1.26 | 1.58 | 0.11 |
| Movimientos estratégicos P1 (%) | 7.91 | 11.28 | 11.98 | 10.39 | 7.88 | 10.05 | 12.80 | 10.24 | -0.15 |
| Victorias IA (2nd) | 0.6392 | 0.6789 | 0.6235 | 0.6472 | 0.6116 | 0.6615 | 0.4440 | 0.5724 | -0.0748 |
| Victorias IA (2nd) | 0.1611 | 0.1625 | 0.1914 | 0.1717 | 0.0898 | 0.1887 | 0.0769 | 0.1185 | -0.0532 |
| Victorias IA (2nd) | 0.0896 | 0.0679 | 0.1242 | 0.0939 | 0.0483 | 0.1264 | 0.0537 | 0.0761 | -0.0178 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

El entrenamiento mejora contra la IA pero empeora contra los rusher.

Evaluar con ALPHA = 0.4 o evaluar con rusher 0.3 y rusher train steps

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_Eval_ALPHA_0_6_RUSHER\s42`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_Eval_ALPHA_0_6_RUSHER\s43`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_Eval_ALPHA_0_6_RUSHER\s44`
