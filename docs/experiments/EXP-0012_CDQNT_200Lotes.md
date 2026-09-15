# EXP-0012 — CDQNT_200Lotes

- **Fecha:** 2026-09-14
- **Veredicto:** positivo
- **Seeds:** 42, 43, 44
- **Lotes por seed:** 200
- **N (partidas paralelas):** 2048
- **Baseline:** models\IAV2_BASE_200

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

| Métrica | BL_s42 | BL_s43 | BL_s44 | BL_μ | NW_s42 | NW_s43 | NW_s44 | NW_μ | Δμ |
|---|---|---|---|---|---|---|---|---|---|
| Win ratio P1 (sin empates) | 52.53 | 69.53 | 49.17 | 57.08 | 56.82 | 84.17 | 49.84 | 63.61 | 6.53 |
| Turnos medios por partida | 9.10 | 8.97 | 7.59 | 8.55 | 8.68 | 9.11 | 8.50 | 8.76 | 0.21 |
| Elo P1 | 912.5 | 1016.8 | 922.6 | 950.6 | 933.9 | 1001.6 | 923.8 | 953.1 | 2.5 |
| Elo P2 | 922.2 | 906.1 | 913.5 | 913.9 | 934.8 | 914.1 | 912.4 | 920.4 | 6.5 |
| Daño medio P1 | 97.56 | 101.02 | 92.69 | 97.09 | 99.37 | 108.57 | 95.99 | 101.31 | 4.22 |
| Reward media P1 | -0.54 | 3.28 | -0.84 | 0.63 | 0.64 | 6.52 | -0.94 | 2.07 | 1.44 |
| Bajas medias P1 | 1.85 | 1.43 | 1.90 | 1.73 | 1.75 | 1.07 | 1.94 | 1.59 | -0.14 |
| Daño por sobrekill medio P1 | 6.11 | 7.29 | 5.63 | 6.34 | 6.61 | 8.19 | 6.15 | 6.98 | 0.64 |
| Kill confirmed medio P1 | 1.87 | 2.24 | 1.73 | 1.95 | 1.99 | 2.51 | 1.89 | 2.13 | 0.18 |
| Defensas desperdiciadas P1 | 1.37 | 1.57 | 1.05 | 1.33 | 1.41 | 1.79 | 1.21 | 1.47 | 0.14 |
| Movimientos estratégicos P1 (%) | 7.61 | 10.13 | 11.90 | 9.88 | 7.91 | 11.28 | 11.98 | 10.39 | 0.51 |
| Victorias IA (2nd) | 0.6429 | 0.6133 | 0.5068 | 0.5877 | 0.6392 | 0.6789 | 0.6235 | 0.6472 | 0.0595 |
| Victorias IA (2nd) | 0.1324 | 0.1525 | 0.1392 | 0.1414 | 0.1611 | 0.1625 | 0.1914 | 0.1717 | 0.0303 |
| Victorias IA (2nd) | 0.0662 | 0.0597 | 0.1199 | 0.0819 | 0.0896 | 0.0679 | 0.1242 | 0.0939 | 0.0120 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

CDQNT a 150 da resultados positivos, por lo que la bueva base incluirá COPY DQN TURN 150.

Evaluacion de CDQN TURN con RUSHER

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNT_200Lotes\s42`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNT_200Lotes\s43`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNT_200Lotes\s44`
