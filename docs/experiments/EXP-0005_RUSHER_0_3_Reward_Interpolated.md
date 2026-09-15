# EXP-0005 — RUSHER_0.3_Reward_Interpolated

- **Fecha:** 2026-09-13
- **Veredicto:** positivo
- **Seeds:** 42, 43, 44
- **Lotes por seed:** 100
- **N (partidas paralelas):** 2048
- **Baseline:** models\IAV2_RUSHER_0_3

## Objetivo / hipótesis

BASE pero con enemigos rusher como posibilidades, quiero estudiar si el cambio en la memoria mejora los resultados con la reward interpolada

El cambio en la memoria de turnos sumado a la interpolaciÃ³nde resultados mejora el aprendizaje con rusher

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
| Win ratio P1 (sin empates) | 28.19 | 80.72 | 57.15 | 55.35 | 59.40 | 80.05 | 45.41 | 61.62 | 6.27 |
| Turnos medios por partida | 8.13 | 9.36 | 9.19 | 8.89 | 9.11 | 9.28 | 7.86 | 8.75 | -0.14 |
| Elo P1 | 951.9 | 1013.4 | 960.7 | 975.3 | 976.0 | 1007.3 | 957.3 | 980.2 | 4.9 |
| Elo P2 | 991.4 | 944.7 | 938.5 | 958.2 | 965.0 | 942.9 | 927.6 | 945.2 | -13.0 |
| Daño medio P1 | 75.35 | 103.36 | 99.93 | 92.88 | 97.84 | 104.45 | 92.54 | 98.28 | 5.40 |
| Reward media P1 | -5.89 | 5.67 | 0.51 | 0.10 | 0.94 | 5.63 | -1.66 | 1.64 | 1.54 |
| Bajas medias P1 | 2.39 | 1.04 | 1.78 | 1.74 | 1.79 | 1.03 | 1.98 | 1.60 | -0.14 |
| Daño por sobrekill medio P1 | 3.67 | 8.35 | 6.42 | 6.15 | 6.26 | 8.21 | 5.48 | 6.65 | 0.50 |
| Kill confirmed medio P1 | 1.13 | 2.53 | 1.96 | 1.87 | 1.91 | 2.50 | 1.68 | 2.03 | 0.16 |
| Defensas desperdiciadas P1 | 0.94 | 1.88 | 1.43 | 1.42 | 1.39 | 1.96 | 1.11 | 1.49 | 0.07 |
| Movimientos estratégicos P1 (%) | 7.50 | 10.80 | 11.30 | 9.87 | 7.21 | 10.54 | 11.23 | 9.66 | -0.21 |
| Victorias IA (2nd) | 0.4647 | 0.6423 | 0.5806 | 0.5626 | 0.6049 | 0.6334 | 0.4005 | 0.5462 | -0.0163 |
| Victorias IA (2nd) | 0.0079 | 0.1908 | 0.1123 | 0.1037 | 0.0427 | 0.1702 | 0.0735 | 0.0954 | -0.0082 |
| Victorias IA (2nd) | 0.0017 | 0.1174 | 0.0790 | 0.0660 | 0.0372 | 0.1043 | 0.0515 | 0.0644 | -0.0016 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

El finetune com la recompensa interpolada parece solucionar el colapso de una de las seeds pero sigue ofreciendo resultados decepcionantes

revisar con fineturne integrado

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_RUSHER_0_3_Reward_Interpolated\s42`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_RUSHER_0_3_Reward_Interpolated\s43`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_RUSHER_0_3_Reward_Interpolated\s44`
