# EXP-0006 — RUSHER_0.3_Reward_Interpolated_FT

- **Fecha:** 2026-09-13
- **Veredicto:** neutro
- **Seeds:** 42, 43, 44
- **Lotes por seed:** 100
- **N (partidas paralelas):** 2048
- **Baseline:** models\IAV2_RUSHER_0_3

## Objetivo / hipótesis

BASE pero con enemigos rusher como posibilidades y epsioidos de entrenamiento contra rusher, ademas de la interpolaciÃ³n de recompensas

El finetune con la memoria de turnos mejora los reusltados del aprendizaje

## Config exacta (diff vs. constants.py por defecto)

| Constante | Valor por defecto | Valor en este experimento |
|---|---|---|
| `DELETE_DIRECTORIES` | `True` | `False` |
| `EVAL_EPISODES` | `2` | `20` |
| `PROFILE_CPROFILE` | `True` | `False` |
| `PROFILE_CPROFILE_OUTPUT` | `profile_cpu.prof` | `optimizedAll2.prof` |
| `RUN_RUSHER_FINETUNE` | `False` | `True` |
| `RUSHER_FINETUNE_EPISODES` | `300` | `100` |
| `RUSHER_TEST_EPISODES` | `100` | `20` |
| `RUSHER_TEST_EPSILON` | `0.05` | `0.0` |
| `TRAIN_EPISODES` | `20` | `100` |
| `USE_WANDB` | `True` | `False` |

## Resultado

| Métrica | BL_s42 | BL_s43 | BL_s44 | BL_μ | NW_s42 | NW_s43 | NW_s44 | NW_μ | Δμ |
|---|---|---|---|---|---|---|---|---|---|
| Win ratio P1 (sin empates) | 28.19 | 80.72 | 57.15 | 55.35 | 36.69 | 80.49 | 40.38 | 52.52 | -2.83 |
| Turnos medios por partida | 8.13 | 9.36 | 9.19 | 8.89 | 8.90 | 9.33 | 8.41 | 8.88 | -0.01 |
| Elo P1 | 951.9 | 1013.4 | 960.7 | 975.3 | 976.0 | 1007.3 | 957.3 | 980.2 | 4.9 |
| Elo P2 | 991.4 | 944.7 | 938.5 | 958.2 | 965.0 | 942.9 | 927.6 | 945.2 | -13.0 |
| Daño medio P1 | 75.35 | 103.36 | 99.93 | 92.88 | 81.58 | 103.88 | 91.36 | 92.27 | -0.61 |
| Reward media P1 | -5.89 | 5.67 | 0.51 | 0.10 | -4.11 | 5.69 | -3.00 | -0.47 | -0.57 |
| Bajas medias P1 | 2.39 | 1.04 | 1.78 | 1.74 | 2.28 | 1.01 | 2.15 | 1.81 | 0.08 |
| Daño por sobrekill medio P1 | 3.67 | 8.35 | 6.42 | 6.15 | 4.22 | 8.17 | 5.00 | 5.80 | -0.35 |
| Kill confirmed medio P1 | 1.13 | 2.53 | 1.96 | 1.87 | 1.27 | 2.46 | 1.53 | 1.75 | -0.12 |
| Defensas desperdiciadas P1 | 0.94 | 1.88 | 1.43 | 1.42 | 1.09 | 1.97 | 1.10 | 1.39 | -0.03 |
| Movimientos estratégicos P1 (%) | 7.50 | 10.80 | 11.30 | 9.87 | 8.63 | 11.33 | 11.98 | 10.65 | 0.78 |
| Victorias IA (2nd) | 0.4647 | 0.6423 | 0.5806 | 0.5626 | 0.3227 | 0.6895 | 0.4783 | 0.4968 | -0.0657 |
| Victorias IA (2nd) | 0.0079 | 0.1908 | 0.1123 | 0.1037 | 0.0039 | 0.1755 | 0.0664 | 0.0820 | -0.0217 |
| Victorias IA (2nd) | 0.0017 | 0.1174 | 0.0790 | 0.0660 | 0.0011 | 0.0973 | 0.0361 | 0.0448 | -0.0212 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

El finetune propio de arquitecturas rusher empeora los resultados del aprendizaje de rusher solo con porcentaje

revisar con fineturne integrado

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_RUSHER_0_3_Reward_Interpolated_FT\s42`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_RUSHER_0_3_Reward_Interpolated_FT\s43`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_RUSHER_0_3_Reward_Interpolated_FT\s44`
