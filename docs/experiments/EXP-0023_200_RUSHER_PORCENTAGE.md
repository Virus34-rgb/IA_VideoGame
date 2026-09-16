# EXP-0023 — 200_RUSHER_PORCENTAGE

- **Fecha:** 2026-09-16
- **Veredicto:** neutro
- **Seeds:** 42, 43, 44
- **Lotes por seed:** 200
- **N (partidas paralelas):** 2048
- **Baseline:** models\IAV3_BASE_200

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
| `RUSHER_FINETUNE_EPISODES` | `300` | `50` |
| `RUSHER_TEST_EPISODES` | `100` | `20` |
| `RUSHER_TEST_EPSILON` | `0.05` | `0.0` |
| `TRAIN_EPISODES` | `20` | `200` |
| `USE_WANDB` | `True` | `False` |

## Resultado

| Métrica | BL_s42 | BL_s43 | BL_s44 | BL_μ | NW_s42 | NW_s43 | NW_s44 | NW_μ | Δμ |
|---|---|---|---|---|---|---|---|---|---|
| Win ratio P1 (sin empates) | 51.99 | 54.88 | 51.81 | 52.89 | 54.95 | 55.14 | 49.21 | 53.10 | 0.21 |
| Turnos medios por partida | 11.24 | 11.07 | 11.27 | 11.19 | 11.40 | 11.05 | 11.57 | 11.34 | 0.15 |
| Elo P1 | 1029.7 | 1038.8 | 1035.9 | 1034.8 | 1024.5 | 1032.1 | 1024.6 | 1027.1 | -7.7 |
| Elo P2 | 1022.5 | 1012.0 | 1021.6 | 1018.7 | 1019.1 | 1012.5 | 1019.7 | 1017.1 | -1.6 |
| Daño medio P1 | 105.34 | 103.19 | 106.83 | 105.12 | 107.79 | 104.79 | 103.32 | 105.30 | 0.18 |
| Reward media P1 | -1.11 | -0.51 | -1.13 | -0.92 | -0.51 | -0.45 | -1.86 | -0.94 | -0.02 |
| Bajas medias P1 | 1.89 | 1.91 | 1.90 | 1.90 | 1.84 | 1.87 | 1.94 | 1.88 | -0.02 |
| Daño por sobrekill medio P1 | 6.40 | 6.44 | 6.27 | 6.37 | 6.50 | 6.59 | 6.15 | 6.41 | 0.04 |
| Kill confirmed medio P1 | 1.96 | 1.99 | 1.96 | 1.97 | 2.00 | 2.03 | 1.90 | 1.98 | 0.01 |
| Defensas desperdiciadas P1 | 1.89 | 1.81 | 1.93 | 1.88 | 1.91 | 1.80 | 1.91 | 1.87 | -0.00 |
| Movimientos estratégicos P1 (%) | 9.58 | 9.28 | 9.32 | 9.39 | 10.05 | 10.40 | 9.27 | 9.91 | 0.51 |
| Victorias IA (2nd) | 0.7066 | 0.6391 | 0.7111 | 0.6856 | 0.7628 | 0.7313 | 0.6958 | 0.7299 | 0.0443 |
| Victorias IA (2nd) | 0.2245 | 0.1887 | 0.2337 | 0.2156 | 0.2529 | 0.2592 | 0.1995 | 0.2372 | 0.0216 |
| Victorias IA (2nd) | 0.1747 | 0.1424 | 0.1510 | 0.1561 | 0.1683 | 0.1712 | 0.1207 | 0.1534 | -0.0027 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

Parece que usar rusher_porcentage = 0.3, trae leves pero mejores resultados.

Evaluar con RUSHER_FINETUNE -> 50 Lotes + 20 fine tune

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV3_200_RUSHER_PORCENTAGE\s42`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV3_200_RUSHER_PORCENTAGE\s43`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV3_200_RUSHER_PORCENTAGE\s44`
