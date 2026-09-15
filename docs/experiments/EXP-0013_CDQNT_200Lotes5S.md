# EXP-0013 — CDQNT_200Lotes5S

- **Fecha:** 2026-09-14
- **Veredicto:** negativo
- **Seeds:** 1, 7, 40, 41, 42
- **Lotes por seed:** 200
- **N (partidas paralelas):** 2048
- **Baseline:** models\IAV2_CDQNT_200Lotes

## Objetivo / hipótesis

Base con CDQNT 150 200 lotes

CDQN Turn en 200 lotes sigue dando buenos resultados 

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

| Métrica | BL_s42 | BL_s43 | BL_s44 | BL_μ | NW_s1 | NW_s7 | NW_s40 | NW_s41 | NW_s42 | NW_μ | Δμ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Win ratio P1 (sin empates) | 56.82 | 84.17 | 49.84 | 63.61 | 44.68 | 71.41 | 59.45 | 28.69 | 56.82 | 52.21 | -11.40 |
| Turnos medios por partida | 8.68 | 9.11 | 8.50 | 8.76 | 9.88 | 7.41 | 10.69 | 8.78 | 8.68 | 9.09 | 0.32 |
| Elo P1 | 933.9 | 1001.6 | 923.8 | 953.1 | 918.6 | 1068.6 | 920.5 | 933.9 | 933.9 | 955.1 | 2.0 |
| Elo P2 | 934.8 | 914.1 | 912.4 | 920.4 | 899.5 | 972.3 | 928.6 | 926.6 | 934.8 | 932.4 | 11.9 |
| Daño medio P1 | 99.37 | 108.57 | 95.99 | 101.31 | 95.64 | 96.27 | 102.99 | 83.24 | 99.37 | 95.50 | -5.81 |
| Reward media P1 | 0.64 | 6.52 | -0.94 | 2.07 | -2.39 | 4.18 | 0.58 | -5.76 | 0.64 | -0.55 | -2.62 |
| Bajas medias P1 | 1.75 | 1.07 | 1.94 | 1.59 | 2.10 | 1.14 | 1.80 | 2.49 | 1.75 | 1.86 | 0.27 |
| Daño por sobrekill medio P1 | 6.61 | 8.19 | 6.15 | 6.98 | 6.03 | 7.59 | 6.69 | 4.21 | 6.61 | 6.23 | -0.76 |
| Kill confirmed medio P1 | 1.99 | 2.51 | 1.89 | 2.13 | 1.82 | 2.26 | 2.04 | 1.28 | 1.99 | 1.88 | -0.25 |
| Defensas desperdiciadas P1 | 1.41 | 1.79 | 1.21 | 1.47 | 1.54 | 1.55 | 1.73 | 1.09 | 1.41 | 1.46 | -0.01 |
| Movimientos estratégicos P1 (%) | 7.91 | 11.28 | 11.98 | 10.39 | 12.85 | 11.82 | 10.65 | 11.03 | 7.91 | 10.85 | 0.46 |
| Victorias IA (2nd) | 0.6392 | 0.6789 | 0.6235 | 0.6472 | 0.5290 | 0.5094 | 0.6524 | 0.4699 | 0.6392 | 0.5600 | -0.0872 |
| Victorias IA (2nd) | 0.1611 | 0.1625 | 0.1914 | 0.1717 | 0.1454 | 0.1012 | 0.1740 | 0.0740 | 0.1611 | 0.1311 | -0.0405 |
| Victorias IA (2nd) | 0.0896 | 0.0679 | 0.1242 | 0.0939 | 0.0932 | 0.0803 | 0.1050 | 0.0421 | 0.0896 | 0.0820 | -0.0119 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

Resultados demasiado discordatnes, queda descartado COPY DQN SEL a 120 y 240.

Evaluacion de CDQN TURN con RUSHER

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNT_200Lotes5S\s1`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNT_200Lotes5S\s7`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNT_200Lotes5S\s40`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNT_200Lotes5S\s41`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNT_200Lotes5S\s42`
