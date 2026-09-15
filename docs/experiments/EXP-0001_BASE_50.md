# EXP-0001 — BASE_50

- **Fecha:** 2026-09-13
- **Veredicto:** referencia
- **Seeds:** 42, 43, 44
- **Lotes por seed:** 50
- **N (partidas paralelas):** 2048
- **Baseline:** (ninguno, referencia)

## Objetivo / hipótesis

Buffer 200k con SAVE Y POOL EVERY fijos cada 10 y 5

Save every no fijo provocaba diferencias entre la relaciÃ³n entre 50-100-200

## Config exacta (diff vs. constants.py por defecto)

| Constante | Valor por defecto | Valor en este experimento |
|---|---|---|
| `DELETE_DIRECTORIES` | `True` | `False` |
| `EVAL_EPISODES` | `2` | `10` |
| `PROFILE_CPROFILE` | `True` | `False` |
| `PROFILE_CPROFILE_OUTPUT` | `profile_cpu.prof` | `optimizedAll2.prof` |
| `RUSHER_FINETUNE_EPISODES` | `300` | `5` |
| `RUSHER_OPPONENT_PERCENTAGE` | `0.3` | `0.0` |
| `RUSHER_TEST_EPISODES` | `100` | `10` |
| `RUSHER_TEST_EPSILON` | `0.05` | `0.0` |
| `TRAIN_EPISODES` | `20` | `50` |
| `USE_WANDB` | `True` | `False` |

## Resultado

| Métrica | NW_s42 | NW_s43 | NW_s44 | NW_μ |
|---|---|---|---|---|
| Win ratio P1 (sin empates) | 56.54 | 70.33 | 53.40 | 60.09 |
| Turnos medios por partida | 9.28 | 8.96 | 7.50 | 8.58 |
| Elo P1 | 1001.3 | 1041.2 | 984.6 | 1009.0 |
| Elo P2 | 967.8 | 952.7 | 943.6 | 954.7 |
| Daño medio P1 | 100.85 | 103.00 | 99.20 | 101.02 |
| Reward media P1 | 0.47 | 3.51 | 0.26 | 1.41 |
| Bajas medias P1 | 1.75 | 1.33 | 1.79 | 1.62 |
| Daño por sobrekill medio P1 | 6.51 | 7.21 | 6.27 | 6.66 |
| Kill confirmed medio P1 | 1.99 | 2.25 | 1.92 | 2.05 |
| Defensas desperdiciadas P1 | 1.56 | 1.52 | 1.26 | 1.45 |
| Movimientos estratégicos P1 (%) | 7.84 | 9.99 | 10.83 | 9.55 |
| Victorias IA (2nd) | 0.6022 | 0.6221 | 0.5995 | 0.6079 |
| Victorias IA (2nd) | 0.1345 | 0.1666 | 0.1558 | 0.1523 |
| Victorias IA (2nd) | 0.0855 | 0.1014 | 0.1051 | 0.0974 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

_(pendiente -- rellenar EXPERIMENT_CONCLUSION)_

confirmar con seeds 7 y 91

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_BASE_50\s42`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_BASE_50\s43`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_BASE_50\s44`
