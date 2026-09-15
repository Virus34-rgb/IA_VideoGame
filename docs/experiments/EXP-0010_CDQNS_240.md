# EXP-0010 — CDQNS_240

- **Fecha:** 2026-09-14
- **Veredicto:** negativo
- **Seeds:** 42, 43, 44
- **Lotes por seed:** 100
- **N (partidas paralelas):** 2048
- **Baseline:** models\IAV2_BASE_100

## Objetivo / hipótesis

Base con CDQN Sel 240

CDQN Sel 240 mejor a el resultado combiando con la memoria

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
| Win ratio P1 (sin empates) | 68.25 | 78.66 | 48.20 | 65.04 | 46.17 | 78.56 | 53.69 | 59.47 | -5.56 |
| Turnos medios por partida | 9.18 | 8.60 | 7.59 | 8.46 | 8.64 | 9.02 | 7.50 | 8.39 | -0.07 |
| Elo P1 | 979.4 | 1018.2 | 958.1 | 985.2 | 953.4 | 1018.7 | 956.8 | 976.3 | -8.9 |
| Elo P2 | 940.4 | 934.3 | 943.2 | 939.3 | 958.2 | 936.0 | 921.8 | 938.7 | -0.6 |
| Daño medio P1 | 104.96 | 104.76 | 96.44 | 102.05 | 89.65 | 106.22 | 92.59 | 96.15 | -5.90 |
| Reward media P1 | 2.98 | 5.50 | -0.92 | 2.52 | -1.90 | 5.33 | 0.16 | 1.20 | -1.32 |
| Bajas medias P1 | 1.49 | 1.06 | 1.91 | 1.49 | 2.04 | 1.13 | 1.78 | 1.65 | 0.16 |
| Daño por sobrekill medio P1 | 7.34 | 8.05 | 5.94 | 7.11 | 5.28 | 7.96 | 6.42 | 6.55 | -0.56 |
| Kill confirmed medio P1 | 2.24 | 2.49 | 1.83 | 2.19 | 1.61 | 2.45 | 1.93 | 2.00 | -0.19 |
| Defensas desperdiciadas P1 | 1.56 | 1.67 | 1.13 | 1.45 | 1.17 | 1.71 | 1.22 | 1.37 | -0.09 |
| Movimientos estratégicos P1 (%) | 9.11 | 11.74 | 12.27 | 11.04 | 8.51 | 11.30 | 12.17 | 10.66 | -0.38 |
| Victorias IA (2nd) | 0.6879 | 0.5901 | 0.5472 | 0.6084 | 0.5523 | 0.6590 | 0.5601 | 0.5905 | -0.0179 |
| Victorias IA (2nd) | 0.1890 | 0.1515 | 0.1395 | 0.1600 | 0.0582 | 0.1755 | 0.1486 | 0.1274 | -0.0326 |
| Victorias IA (2nd) | 0.1146 | 0.0898 | 0.1008 | 0.1017 | 0.0342 | 0.0941 | 0.1040 | 0.0774 | -0.0243 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

Fracaso, descartado aumentar el DQN SEL a 150 o 300

Evaluacion de CDQN_SEL y DQN_TURN

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNS_240\s42`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNS_240\s43`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNS_240\s44`
