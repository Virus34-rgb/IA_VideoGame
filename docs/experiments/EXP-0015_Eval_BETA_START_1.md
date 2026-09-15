# EXP-0015 — Eval_BETA_START_1

- **Fecha:** 2026-09-15
- **Veredicto:** positivo
- **Seeds:** 42, 43, 44
- **Lotes por seed:** 200
- **N (partidas paralelas):** 2048
- **Baseline:** models\IAV2_BASE_200

## Objetivo / hipótesis

BASE con BETA_START = 1.0 

La varianza del PER inicial distorisona el aprendizaje 

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
| Win ratio P1 (sin empates) | 56.82 | 84.17 | 49.84 | 63.61 | 58.06 | 82.93 | 59.00 | 66.66 | 3.05 |
| Turnos medios por partida | 8.68 | 9.11 | 8.50 | 8.76 | 8.88 | 9.54 | 7.89 | 8.77 | 0.01 |
| Elo P1 | 933.9 | 1001.6 | 923.8 | 953.1 | 942.6 | 989.8 | 924.8 | 952.4 | -0.7 |
| Elo P2 | 934.8 | 914.1 | 912.4 | 920.4 | 946.2 | 920.3 | 879.3 | 915.3 | -5.2 |
| Daño medio P1 | 99.37 | 108.57 | 95.99 | 101.31 | 96.48 | 106.88 | 95.72 | 99.69 | -1.62 |
| Reward media P1 | 0.64 | 6.52 | -0.94 | 2.07 | 0.73 | 6.09 | 1.19 | 2.67 | 0.60 |
| Bajas medias P1 | 1.75 | 1.07 | 1.94 | 1.59 | 1.66 | 1.07 | 1.69 | 1.47 | -0.11 |
| Daño por sobrekill medio P1 | 6.61 | 8.19 | 6.15 | 6.98 | 6.45 | 8.40 | 6.48 | 7.11 | 0.13 |
| Kill confirmed medio P1 | 1.99 | 2.51 | 1.89 | 2.13 | 1.96 | 2.52 | 1.94 | 2.14 | 0.01 |
| Defensas desperdiciadas P1 | 1.41 | 1.79 | 1.21 | 1.47 | 1.40 | 1.84 | 1.25 | 1.50 | 0.03 |
| Movimientos estratégicos P1 (%) | 7.91 | 11.28 | 11.98 | 10.39 | 7.57 | 11.52 | 11.50 | 10.20 | -0.19 |
| Victorias IA (2nd) | 0.6392 | 0.6789 | 0.6235 | 0.6472 | 0.5797 | 0.6928 | 0.6259 | 0.6328 | -0.0144 |
| Victorias IA (2nd) | 0.1611 | 0.1625 | 0.1914 | 0.1717 | 0.0841 | 0.1952 | 0.1932 | 0.1575 | -0.0141 |
| Victorias IA (2nd) | 0.0896 | 0.0679 | 0.1242 | 0.0939 | 0.0580 | 0.1018 | 0.1274 | 0.0957 | 0.0018 |

## Confusores conocidos

_(ninguno -- rellenar EXPERIMENT_CONFUSORES)_

## Conclusión y siguiente paso

Los resultados del analisis son confusos, trae una clara mejora en winrate contra la IA del selfplay pero pierde levemente contra rusher leve y medio, mientras que sube con rusher agresivo. No llego a ninguna conclusión.

Evaluar con BETA_START = 0.6

## Nota

_(sin notas)_

## Artefactos (carpetas por seed)

- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_Eval_BETA_START_1\s42`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_Eval_BETA_START_1\s43`
- `C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_Eval_BETA_START_1\s44`
