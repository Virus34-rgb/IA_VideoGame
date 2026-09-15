# EXP-0008 — CDQNT_300

- **Fecha:** 2026-09-13
- **Veredicto:** negativo
- **Seeds:** 42, 43, 44
- **Lotes por seed:** 100
- **N (partidas paralelas):** 2048
- **Baseline:** models\IAV2_CDQNT_150

## Objetivo / hipótesis

Base con CDQN Turn 300

CDQN Turn 300 mejor a el resultado combiando con la memoria

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
| Win ratio P1 (sin empates) | 63.17 | 84.79 | 63.57 | 70.51 | 65.44 | 83.46 | 47.78 | 65.56 | -4.95 |
| Turnos medios por partida | 7.92 | 9.36 | 8.09 | 8.46 | 8.17 | 8.64 | 7.35 | 8.05 | -0.40 |
| Elo P1 | 966.5 | 1043.4 | 986.6 | 998.8 | 954.4 | 1026.7 | 981.9 | 987.7 | -11.2 |
| Elo P2 | 952.0 | 934.1 | 931.6 | 939.2 | 966.0 | 949.1 | 945.3 | 953.5 | 14.2 |
| Daño medio P1 | 96.66 | 108.01 | 102.56 | 102.41 | 98.77 | 105.38 | 93.75 | 99.30 | -3.11 |
| Reward media P1 | 2.18 | 6.62 | 2.26 | 3.69 | 2.59 | 6.51 | -1.01 | 2.70 | -0.99 |
| Bajas medias P1 | 1.46 | 0.92 | 1.54 | 1.31 | 1.47 | 0.92 | 1.90 | 1.43 | 0.12 |
| Daño por sobrekill medio P1 | 6.57 | 8.46 | 7.01 | 7.35 | 6.68 | 8.31 | 6.05 | 7.01 | -0.33 |
| Kill confirmed medio P1 | 1.99 | 2.59 | 2.15 | 2.24 | 2.03 | 2.55 | 1.84 | 2.14 | -0.10 |
| Defensas desperdiciadas P1 | 1.35 | 1.98 | 1.40 | 1.58 | 1.39 | 1.78 | 1.13 | 1.43 | -0.14 |
| Movimientos estratégicos P1 (%) | 8.65 | 10.61 | 12.97 | 10.74 | 10.07 | 11.19 | 11.59 | 10.95 | 0.21 |
| Victorias IA (2nd) | 0.5683 | 0.6820 | 0.6485 | 0.6329 | 0.6371 | 0.6465 | 0.5773 | 0.6203 | -0.0126 |
| Victorias IA (2nd) | 0.1388 | 0.1911 | 0.1799 | 0.1699 | 0.1697 | 0.1762 | 0.1391 | 0.1617 | -0.0083 |
| Victorias IA (2nd) | 0.0918 | 0.1094 | 0.1177 | 0.1063 | 0.1172 | 0.0948 | 0.0943 | 0.1021 | -0.0042 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

CDQN_TURN 300 da peores resultados que CDQNT 150, queda descartado el valor de 300.

Evaluacion de CDQN_SEL 120 y 240

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNT_300\s42`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNT_300\s43`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNT_300\s44`
