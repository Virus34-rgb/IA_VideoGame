# EXP-0025 — 100_RUSHER_FINETUNE_40_LOTES

- **Fecha:** 2026-09-16
- **Veredicto:** negativo
- **Seeds:** 42, 43, 44
- **Lotes por seed:** 100
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
| `RUSHER_FINETUNE_EPISODES` | `300` | `40` |
| `RUSHER_TEST_EPISODES` | `100` | `20` |
| `RUSHER_TEST_EPSILON` | `0.05` | `0.0` |
| `TRAIN_EPISODES` | `20` | `100` |
| `USE_WANDB` | `True` | `False` |

## Resultado

| Métrica | BL_s42 | BL_s43 | BL_s44 | BL_μ | NW_s42 | NW_s43 | NW_s44 | NW_μ | Δμ |
|---|---|---|---|---|---|---|---|---|---|
| Win ratio P1 (sin empates) | 53.52 | 60.40 | 48.85 | 54.26 | 43.10 | 37.07 | 43.56 | 41.24 | -13.01 |
| Turnos medios por partida | 10.97 | 11.06 | 11.61 | 11.21 | 6.59 | 6.76 | 6.34 | 6.56 | -4.65 |
| Elo P1 | 1026.4 | 1027.5 | 1009.1 | 1021.0 | 983.7 | 987.2 | 996.1 | 989.0 | -32.0 |
| Elo P2 | 1010.2 | 996.2 | 1013.9 | 1006.8 | 982.9 | 990.9 | 1013.3 | 995.7 | -11.1 |
| Daño medio P1 | 104.51 | 105.61 | 102.98 | 104.37 | 90.14 | 87.64 | 88.23 | 88.67 | -15.70 |
| Reward media P1 | -0.79 | 0.70 | -1.95 | -0.68 | -1.90 | -3.33 | -1.83 | -2.35 | -1.67 |
| Bajas medias P1 | 1.89 | 1.80 | 1.95 | 1.88 | 2.06 | 2.17 | 2.06 | 2.10 | 0.22 |
| Daño por sobrekill medio P1 | 6.40 | 6.72 | 6.16 | 6.43 | 5.62 | 5.45 | 5.48 | 5.52 | -0.91 |
| Kill confirmed medio P1 | 1.97 | 2.05 | 1.90 | 1.97 | 1.72 | 1.67 | 1.67 | 1.69 | -0.29 |
| Defensas desperdiciadas P1 | 1.76 | 1.92 | 1.91 | 1.86 | 0.86 | 0.88 | 0.79 | 0.84 | -1.02 |
| Movimientos estratégicos P1 (%) | 8.52 | 7.99 | 9.21 | 8.57 | 7.11 | 8.32 | 6.43 | 7.29 | -1.29 |
| Victorias IA (2nd) | 0.7269 | 0.7097 | 0.6783 | 0.7050 | 0.6090 | 0.6278 | 0.6358 | 0.6242 | -0.0808 |
| Victorias IA (2nd) | 0.2464 | 0.2506 | 0.1775 | 0.2248 | 0.1436 | 0.1878 | 0.1658 | 0.1658 | -0.0591 |
| Victorias IA (2nd) | 0.1519 | 0.1571 | 0.1037 | 0.1376 | 0.0727 | 0.1033 | 0.1057 | 0.0939 | -0.0437 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

El orden corrompe los resultado por completo

Evaluar con RUSHER_FINETUNE -> 200 Lotes + 60 fine tune

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV3_100_RUSHER_FINETUNE_40_LOTES\s42`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV3_100_RUSHER_FINETUNE_40_LOTES\s43`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV3_100_RUSHER_FINETUNE_40_LOTES\s44`
