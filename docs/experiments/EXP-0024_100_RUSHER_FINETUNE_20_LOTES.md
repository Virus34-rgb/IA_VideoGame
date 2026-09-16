# EXP-0024 — 100_RUSHER_FINETUNE_20_LOTES

- **Fecha:** 2026-09-16
- **Veredicto:** neutro
- **Seeds:** 42, 43, 44
- **Lotes por seed:** 50
- **N (partidas paralelas):** 2048
- **Baseline:** models\IAV3_50_RUSHER_PORCENTAGE

## Objetivo / hipótesis

Observar si ahora el finetune con rusher ayuda al aprendizaje

El cacheo de las redes noisy es lo que perjudicÃ³ el aprendizaje con rusher finetune 

## Config exacta (diff vs. constants.py por defecto)

| Constante | Valor por defecto | Valor en este experimento |
|---|---|---|
| `DELETE_DIRECTORIES` | `True` | `False` |
| `EVAL_EPISODES` | `2` | `10` |
| `PROFILE_CPROFILE` | `True` | `False` |
| `PROFILE_CPROFILE_OUTPUT` | `profile_cpu.prof` | `optimizedAll2.prof` |
| `RUN_RUSHER_FINETUNE` | `False` | `True` |
| `RUSHER_FINETUNE_EPISODES` | `300` | `20` |
| `RUSHER_TEST_EPISODES` | `100` | `10` |
| `RUSHER_TEST_EPSILON` | `0.05` | `0.0` |
| `TRAIN_EPISODES` | `20` | `50` |
| `USE_WANDB` | `True` | `False` |

## Resultado

| Métrica | BL_s42 | BL_s43 | BL_s44 | BL_μ | NW_s42 | NW_s43 | NW_s44 | NW_μ | Δμ |
|---|---|---|---|---|---|---|---|---|---|
| Win ratio P1 (sin empates) | 53.28 | 55.44 | 49.49 | 52.74 | 51.50 | 52.48 | 50.76 | 51.58 | -1.16 |
| Turnos medios por partida | 11.07 | 11.44 | 11.49 | 11.33 | 11.15 | 11.57 | 11.51 | 11.41 | 0.08 |
| Elo P1 | 1016.6 | 1014.8 | 1013.0 | 1014.8 | 1016.6 | 1014.8 | 1013.0 | 1014.8 | 0.0 |
| Elo P2 | 1008.0 | 1008.9 | 1009.2 | 1008.7 | 1008.0 | 1008.9 | 1009.2 | 1008.7 | 0.0 |
| Daño medio P1 | 104.57 | 107.17 | 102.26 | 104.67 | 104.80 | 106.38 | 103.16 | 104.78 | 0.11 |
| Reward media P1 | -0.87 | -0.43 | -1.79 | -1.03 | -1.22 | -1.11 | -1.58 | -1.30 | -0.27 |
| Bajas medias P1 | 1.86 | 1.80 | 1.92 | 1.86 | 1.89 | 1.87 | 1.90 | 1.89 | 0.03 |
| Daño por sobrekill medio P1 | 6.40 | 6.67 | 6.17 | 6.41 | 6.43 | 6.35 | 6.23 | 6.34 | -0.08 |
| Kill confirmed medio P1 | 1.98 | 2.03 | 1.90 | 1.97 | 1.97 | 1.98 | 1.92 | 1.96 | -0.01 |
| Defensas desperdiciadas P1 | 1.72 | 1.92 | 1.95 | 1.86 | 1.82 | 1.89 | 1.85 | 1.85 | -0.01 |
| Movimientos estratégicos P1 (%) | 7.07 | 7.00 | 7.76 | 7.28 | 7.85 | 7.68 | 7.73 | 7.75 | 0.48 |
| Victorias IA (2nd) | 0.7101 | 0.6534 | 0.7255 | 0.6963 | 0.6806 | 0.6712 | 0.7201 | 0.6906 | -0.0057 |
| Victorias IA (2nd) | 0.2612 | 0.1872 | 0.2555 | 0.2346 | 0.2514 | 0.2148 | 0.2640 | 0.2434 | 0.0087 |
| Victorias IA (2nd) | 0.1669 | 0.1222 | 0.1732 | 0.1541 | 0.1467 | 0.1435 | 0.1668 | 0.1523 | -0.0018 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

_(pendiente -- rellenar EXPERIMENT_CONCLUSION)_

Evaluar con RUSHER_FINETUNE -> 100 Lotes + 40 fine tune

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV3_100_RUSHER_FINETUNE_20_LOTES\s42`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV3_100_RUSHER_FINETUNE_20_LOTES\s43`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV3_100_RUSHER_FINETUNE_20_LOTES\s44`
