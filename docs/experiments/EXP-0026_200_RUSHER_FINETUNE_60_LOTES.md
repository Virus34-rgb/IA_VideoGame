# EXP-0026 — 200_RUSHER_FINETUNE_60_LOTES

- **Fecha:** 2026-09-16
- **Veredicto:** negativo
- **Seeds:** 42, 43, 44
- **Lotes por seed:** 200
- **N (partidas paralelas):** 2048
- **Baseline:** models\IAV3_100_RUSHER_PORCENTAGE

## Objetivo / hipótesis

Observar si ahora el finetune con rusher ayuda al aprendizaje

El cacheo de las redes noisy es lo que perjudicÃ³ el aprendizaje con rusher finetune 

## Config exacta (diff vs. constants.py por defecto)

| Constante | Valor por defecto | Valor en este experimento |
|---|---|---|
| `DELETE_DIRECTORIES` | `True` | `False` |
| `EVAL_EPISODES` | `2` | `20` |
| `PROFILE_CPROFILE` | `True` | `False` |
| `PROFILE_CPROFILE_OUTPUT` | `profile_cpu.prof` | `optimizedAll2.prof` |
| `RUN_RUSHER_FINETUNE` | `False` | `True` |
| `RUSHER_FINETUNE_EPISODES` | `300` | `60` |
| `RUSHER_TEST_EPISODES` | `100` | `20` |
| `RUSHER_TEST_EPSILON` | `0.05` | `0.0` |
| `TRAIN_EPISODES` | `20` | `200` |
| `USE_WANDB` | `True` | `False` |

## Resultado

| Métrica | BL_s42 | BL_s43 | BL_s44 | BL_μ | NW_s42 | NW_s43 | NW_s44 | NW_μ | Δμ |
|---|---|---|---|---|---|---|---|---|---|
| Win ratio P1 (sin empates) | 53.52 | 60.40 | 48.85 | 54.26 | 42.20 | 41.12 | 55.40 | 46.24 | -8.02 |
| Turnos medios por partida | 10.97 | 11.06 | 11.61 | 11.21 | 6.55 | 6.62 | 6.15 | 6.44 | -4.77 |
| Elo P1 | 1026.4 | 1027.5 | 1009.1 | 1021.0 | 976.5 | 969.8 | 1024.5 | 990.3 | -30.7 |
| Elo P2 | 1010.2 | 996.2 | 1013.9 | 1006.8 | 983.9 | 995.3 | 999.6 | 992.9 | -13.8 |
| Daño medio P1 | 104.51 | 105.61 | 102.98 | 104.37 | 86.88 | 86.64 | 89.08 | 87.53 | -16.83 |
| Reward media P1 | -0.79 | 0.70 | -1.95 | -0.68 | -2.15 | -2.45 | 0.65 | -1.32 | -0.64 |
| Bajas medias P1 | 1.89 | 1.80 | 1.95 | 1.88 | 2.07 | 2.09 | 1.81 | 1.99 | 0.11 |
| Daño por sobrekill medio P1 | 6.40 | 6.72 | 6.16 | 6.43 | 5.69 | 5.60 | 5.72 | 5.67 | -0.76 |
| Kill confirmed medio P1 | 1.97 | 2.05 | 1.90 | 1.97 | 1.73 | 1.70 | 1.73 | 1.72 | -0.25 |
| Defensas desperdiciadas P1 | 1.76 | 1.92 | 1.91 | 1.86 | 0.83 | 0.90 | 0.74 | 0.82 | -1.04 |
| Movimientos estratégicos P1 (%) | 8.52 | 7.99 | 9.21 | 8.57 | 8.25 | 7.64 | 8.27 | 8.05 | -0.52 |
| Victorias IA (2nd) | 0.7269 | 0.7097 | 0.6783 | 0.7050 | 0.5902 | 0.6258 | 0.6096 | 0.6085 | -0.0965 |
| Victorias IA (2nd) | 0.2464 | 0.2506 | 0.1775 | 0.2248 | 0.1413 | 0.1712 | 0.1714 | 0.1613 | -0.0635 |
| Victorias IA (2nd) | 0.1519 | 0.1571 | 0.1037 | 0.1376 | 0.0937 | 0.1126 | 0.1262 | 0.1108 | -0.0267 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

_(pendiente -- rellenar EXPERIMENT_CONCLUSION)_

Evaluar con RUSHER_FINETUNE -> 200 Lotes + 60 fine tune

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV3_200_RUSHER_FINETUNE_60_LOTES\s42`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV3_200_RUSHER_FINETUNE_60_LOTES\s43`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV3_200_RUSHER_FINETUNE_60_LOTES\s44`
