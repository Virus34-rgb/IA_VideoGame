# EXP-0021 — 50_RUSHER_PORCENTAGE

- **Fecha:** 2026-09-16
- **Veredicto:** negativo
- **Seeds:** 42, 43, 44
- **Lotes por seed:** 50
- **N (partidas paralelas):** 2048
- **Baseline:** models\IAV2_BASE_50

## Objetivo / hipótesis

Observar si ahora el procentaje de rusher ayuda al aprendizaje

El cahceo de las redes noisy es lo que perjudicÃ³ el aprendizaje con rusher porcentage 

## Config exacta (diff vs. constants.py por defecto)

| Constante | Valor por defecto | Valor en este experimento |
|---|---|---|
| `DELETE_DIRECTORIES` | `True` | `False` |
| `EVAL_EPISODES` | `2` | `10` |
| `PROFILE_CPROFILE` | `True` | `False` |
| `PROFILE_CPROFILE_OUTPUT` | `profile_cpu.prof` | `optimizedAll2.prof` |
| `RUSHER_FINETUNE_EPISODES` | `300` | `100` |
| `RUSHER_TEST_EPISODES` | `100` | `10` |
| `RUSHER_TEST_EPSILON` | `0.05` | `0.0` |
| `TRAIN_EPISODES` | `20` | `50` |
| `USE_WANDB` | `True` | `False` |

## Resultado

| Métrica | BL_s42 | BL_s43 | BL_s44 | BL_μ | NW_s42 | NW_s43 | NW_s44 | NW_μ | Δμ |
|---|---|---|---|---|---|---|---|---|---|
| Win ratio P1 (sin empates) | 57.18 | 69.80 | 65.13 | 64.04 | 53.28 | 55.44 | 49.49 | 52.74 | -11.30 |
| Turnos medios por partida | 9.09 | 9.91 | 7.31 | 8.77 | 11.07 | 11.44 | 11.49 | 11.33 | 2.56 |
| Elo P1 | 1016.9 | 1052.3 | 1013.9 | 1027.7 | 1016.6 | 1014.8 | 1013.0 | 1014.8 | -12.9 |
| Elo P2 | 969.5 | 950.7 | 943.7 | 954.6 | 1008.0 | 1008.9 | 1009.2 | 1008.7 | 54.1 |
| Daño medio P1 | 98.33 | 103.10 | 98.09 | 99.84 | 104.57 | 107.17 | 102.26 | 104.67 | 4.83 |
| Reward media P1 | 0.56 | 3.06 | 2.73 | 2.12 | -0.87 | -0.43 | -1.79 | -1.03 | -3.15 |
| Bajas medias P1 | 1.73 | 1.39 | 1.48 | 1.53 | 1.86 | 1.80 | 1.92 | 1.86 | 0.33 |
| Daño por sobrekill medio P1 | 6.65 | 7.31 | 6.86 | 6.94 | 6.40 | 6.67 | 6.17 | 6.41 | -0.53 |
| Kill confirmed medio P1 | 2.01 | 2.27 | 2.10 | 2.13 | 1.98 | 2.03 | 1.90 | 1.97 | -0.16 |
| Defensas desperdiciadas P1 | 1.51 | 1.73 | 1.25 | 1.50 | 1.72 | 1.92 | 1.95 | 1.86 | 0.37 |
| Movimientos estratégicos P1 (%) | 9.00 | 10.36 | 9.56 | 9.64 | 7.07 | 7.00 | 7.76 | 7.28 | -2.36 |
| Victorias IA (2nd) | 0.6019 | 0.6629 | 0.6527 | 0.6392 | 0.7101 | 0.6534 | 0.7255 | 0.6963 | 0.0572 |
| Victorias IA (2nd) | 0.1522 | 0.1935 | 0.2069 | 0.1842 | 0.2612 | 0.1872 | 0.2555 | 0.2346 | 0.0504 |
| Victorias IA (2nd) | 0.0773 | 0.1107 | 0.1394 | 0.1091 | 0.1669 | 0.1222 | 0.1732 | 0.1541 | 0.0450 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

Los resultados mo son concluyentes sobre la mejora en un rango tan corto de lotes. Aunque parece que suar rusher_porcentage = 0.3, trae leves pero mejores resultados.

Evaluar con 100 lotes

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV3_50_RUSHER_PORCENTAGE\s42`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV3_50_RUSHER_PORCENTAGE\s43`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV3_50_RUSHER_PORCENTAGE\s44`
