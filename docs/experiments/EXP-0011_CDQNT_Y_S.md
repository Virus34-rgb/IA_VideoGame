# EXP-0011 — CDQNT_Y_S

- **Fecha:** 2026-09-14
- **Veredicto:** negativo
- **Seeds:** 42, 43, 44
- **Lotes por seed:** 100
- **N (partidas paralelas):** 2048
- **Baseline:** models\IAV2_CDQNT_150

## Objetivo / hipótesis

Base con CDQN Sel 12o Y CDQNT 150

CDQN Sel 120 junto a CDQNT dan buenos resultados 

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
| Win ratio P1 (sin empates) | 63.17 | 84.79 | 63.57 | 70.51 | 53.86 | 83.33 | 56.39 | 64.53 | -5.98 |
| Turnos medios por partida | 7.92 | 9.36 | 8.09 | 8.46 | 8.20 | 9.53 | 7.68 | 8.47 | 0.01 |
| Elo P1 | 966.5 | 1043.4 | 986.6 | 998.8 | 938.6 | 1042.3 | 963.3 | 981.4 | -17.4 |
| Elo P2 | 952.0 | 934.1 | 931.6 | 939.2 | 975.4 | 929.1 | 941.8 | 948.8 | 9.5 |
| Daño medio P1 | 96.66 | 108.01 | 102.56 | 102.41 | 91.00 | 107.30 | 99.25 | 99.18 | -3.23 |
| Reward media P1 | 2.18 | 6.62 | 2.26 | 3.69 | -0.05 | 6.23 | 0.85 | 2.34 | -1.34 |
| Bajas medias P1 | 1.46 | 0.92 | 1.54 | 1.31 | 1.68 | 0.97 | 1.70 | 1.45 | 0.14 |
| Daño por sobrekill medio P1 | 6.57 | 8.46 | 7.01 | 7.35 | 5.87 | 8.38 | 6.74 | 7.00 | -0.35 |
| Kill confirmed medio P1 | 1.99 | 2.59 | 2.15 | 2.24 | 1.78 | 2.56 | 2.04 | 2.13 | -0.12 |
| Defensas desperdiciadas P1 | 1.35 | 1.98 | 1.40 | 1.58 | 1.33 | 1.95 | 1.30 | 1.53 | -0.05 |
| Movimientos estratégicos P1 (%) | 8.65 | 10.61 | 12.97 | 10.74 | 9.00 | 11.31 | 12.98 | 11.10 | 0.35 |
| Victorias IA (2nd) | 0.5683 | 0.6820 | 0.6485 | 0.6329 | 0.5578 | 0.6581 | 0.5694 | 0.5951 | -0.0378 |
| Victorias IA (2nd) | 0.1388 | 0.1911 | 0.1799 | 0.1699 | 0.0931 | 0.1784 | 0.1507 | 0.1408 | -0.0292 |
| Victorias IA (2nd) | 0.0918 | 0.1094 | 0.1177 | 0.1063 | 0.0798 | 0.1002 | 0.1031 | 0.0944 | -0.0119 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

Viendo las estadísticas de la conclusión se observa que CDQN SEL a 120 mejora los resultados de 2 seeds y empeora el de 1 seed, pero las magnitdes muestran como el ratio de empeorar es mayor.

Verificar si CDQN_SEL tiene este comportamiento caótico con 5 semillas, de las cuales 4 son diferentes.

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNT_Y_S\s42`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNT_Y_S\s43`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_CDQNT_Y_S\s44`
