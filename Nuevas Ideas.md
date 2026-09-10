Actúa como un Ingeniero Principal de Aprendizaje por Refuerzo (RL) y Especialista en Optimización de Rendimiento en PyTorch. Tu objetivo es auditar un proyecto de RL (DQN) para un videojuego de batallas de estrategia 3v3.

---

## 1. CONTEXTO, IDEAS PREVIAS Y RESULTADOS ACTUALES

### Historial de Ideas Propuestas:
1. Muestreo por bandas ponderadas de agresividad (sobremuestreo de estilos agresivos) - [IMPLEMENTADO]
2. Prior inicial del perfil por composición enemiga - [IMPLEMENTADO]
3. EMA en vez de media acumulada en el perfil - [IMPLEMENTADO]
4. Curriculum de fases (Fases 1/2/3) - [IMPLEMENTADO]
5. Activar `RUN_RUSHER_FINETUNE` - [IMPLEMENTADO]
6. Entrenamiento adaptativo por winrate (estilo Prioritized Level Replay)
7. Reward de riesgo de supervivencia (R = Rnormal − λt·HP_loss − μt·P(death_next_turn))
8. Apertura conservadora fija por nº de turno (sin condicionar al rival)
9. Opponent modeling temprano / perfil con más dimensiones (Señal más rica)
10. Auditoría de recompensa simétrica
11. Ablation de `wasted_defense`
12. Liga con exploiter persistente (estilo AlphaStar)

### Resultados de Métricas tras implementar los puntos 1 al 5:
*   **100 episodios de entrenamiento:**
    *   Vs 0.0 (Pasivo) = 66.94% winrate | 17.46 turnos promedio
    *   Vs 0.5 (Medio) = 19.26% winrate | 8.67 turnos promedio
    *   Vs 1.0 (Agresivo) = 11.95% winrate | 5.81 turnos promedio (10.86 turnos globales)
    *   Vs IA2 = 58.42% winrate (Aprende mucho menos por pooling [9.4] y por rusher [0.3])
*   **500 episodios de entrenamiento:**
    *   Vs 0.0 = 74.19% winrate | 17.40 turnos promedio
    *   Vs 0.5 = 24.91% winrate | 8.53 turnos promedio
    *   Vs 1.0 = 16.80% winrate | 5.66 turnos promedio
    *   Vs IA2 = 57.81% winrate | 10.81 turnos promedio

*Nota actual:* Estoy corriendo un entrenamiento de 2000 episodios para ver si el winrate contra el perfil agresivo (Vs 1.0) se acerca al 40%. Sin embargo, el crecimiento del winrate es notablemente lento.

---

## 2. ESPECIFICACIONES DEL ENTORNO Y RESTRICCIONES
*   **El Problema:** Juego de batallas 3v3 guerreros. 6 decisiones posibles por turno.
*   **Arquitectura:** 2 Redes Neuronales DQN concurrentes (Red A: Selección de guerreros / Red B: Selección de acciones en partida).
*   **Mecánicas actuales de RL:** DoubleDQN, N-Steps, PER, DuelingDQN, OpponentPool, NoisyNetwork, Oponentes de heurística (igual se me olviada algo).
*   **Restricción de Replay Buffer:** No se permite disminuir el tamaño de los replays de las redes a menos que el sistema permita aumentar el tamaño del batch (actualmente fijado en Batch Size = 128).

---

## 3. TAREAS ESTRICTAS A REALIZAR

Analiza exhaustivamente los archivos del proyecto adjuntos y el perfil de entrenamiento para ejecutar las siguientes tres tareas sin omitir ninguna:

### TAREA 1: Análisis de Resultados y Estrategia Antiestancamiento
1. Diagnóstica el crecimiento lento del winrate contra heurísticas agresivas (Vs 0.5 y Vs 1.0). ¿Es el comportamiento esperado con la arquitectura actual?
2. Actualiza la lista de ideas eliminando los 5 puntos ya implementados. 
3. Añade nuevas propuestas avanzadas de RL (mecánicas, modulación de rewards o entornos) diseñadas específicamente para romper el estancamiento contra estrategias "Rusher"/Agresivas.

### TAREA 2: Profiling y Optimización de Rendimiento (Meta: <10 min por 100 eps)
1. Analiza el reporte del Profiler adjunto para identificar cuellos de botella en la ejecución (CPU vs GPU, cuellos de botella en el step del entorno, transferencia de tensores, etc.).
2. Propón mejoras de código y configuraciones de PyTorch/Entorno (ej. `torch.compile`, pinned memory, optimizaciones de recolección de experiencias).
3. Mantén la restricción de que el código optimizado debe priorizar la eficiencia de tiempo sin dañar la convergencia del aprendizaje.

### TAREA 3: Auditoría de Arquitectura de Código (Clean Code e Industria)
1. Evalúa la responsabilidad de cada clase en los archivos del proyecto.
2. Identifica violación de principios SOLID, código duplicado o dependencias acopladas que dificulten la mantenibilidad.
3. Diseña un Plan de Acción estructurado por pasos para refactorizar el proyecto bajo estándares de la industria (Clean Code), garantizando no romper la eficiencia de ejecución ni el proceso de RL.
4. Ten en cuenta todos los archivos del proyecto, y se asume la creación de nuevos archivos.

### TAREA 4: Recomendación de Nuevas Mecánicas de RL
1. Revisa las mecánicas ya implementadas (DoubleDQN, N-Steps,...).
2. Recomienda técnicas adicionales compatibles (que no este implementadas) con el ecosistema DQN (Multi-step Bootstrap Targets, Distributional RL (Categorical/C51 o QR-DQN) o Munchausen RL,etc.) que se adapten al problema 3v3.
3. Evalúa cada recomendación de forma exhaustiva mediante una matriz/análisis de tres variables: **[Impacto en Aprendizaje] vs [Coste de Tiempo de Ejecución] vs [Coste de Implementación]**. No escatimes en opciones; incluye ideas de alto coste si su impacto estimado en el winrate es masivo.

---

## 4. ARCHIVOS Y DATOS ADJUNTOS PARA EL ANÁLISIS

### Perfil de Entrenamiento (100 episodios):
Log de la ejecución:
=================================================================
                    CASTLE GAME (VECTORIZADO)
=================================================================
Versión:  IA V2   |   N (partidas por lote): 2048
  - Entrenamiento self-play: train, 100 lotes
  - Fine-tuning vs Rusher (fase 1, agg 0.0-0.3): evaluate, 15 lotes
  - Fine-tuning vs Rusher (fase 2, agg 0.3-0.6): evaluate, 35 lotes
  - Fine-tuning vs Rusher (fase 3, agg 0.6-1): evaluate, 50 lotes
  - Evaluación final: evaluate, 50 lotes
  - Evaluación vs Rusher (aggression=0.0): evaluate, 50 lotes
  - Evaluación vs Rusher (aggression=0.5): evaluate, 50 lotes
  - Evaluación vs Rusher (aggression=1.0): evaluate, 50 lotes
Logs en:  C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\models\IAV2_Tiempo
Wandb:    Desactivado
=================================================================

=================================================================
STEP: Entrenamiento self-play (train, 100 lotes de 2048)
-----------------------------------------------------------------
[100.0%] Lote 100/100 |    0.1 lotes/s | ETA 0.0s     
Entrenamiento self-play terminado en 24m 31.25s

=================================================================
STEP: Fine-tuning vs Rusher (fase 1, agg 0.0-0.3) (evaluate, 15 lotes de 2048)
-----------------------------------------------------------------
[100.0%] Lote 15/15 |    0.1 lotes/s | ETA 0.0s    
Fine-tuning vs Rusher (fase 1, agg 0.0-0.3) terminado en 2m 5.64s

=================================================================
STEP: Fine-tuning vs Rusher (fase 2, agg 0.3-0.6) (evaluate, 35 lotes de 2048)
-----------------------------------------------------------------
[100.0%] Lote 35/35 |    0.1 lotes/s | ETA 0.0s     
Fine-tuning vs Rusher (fase 2, agg 0.3-0.6) terminado en 4m 44.30s

=================================================================
STEP: Fine-tuning vs Rusher (fase 3, agg 0.6-1) (evaluate, 50 lotes de 2048)
-----------------------------------------------------------------
[100.0%] Lote 50/50 |    0.1 lotes/s | ETA 0.0s     
Fine-tuning vs Rusher (fase 3, agg 0.6-1) terminado en 7m 1.11s

=================================================================
STEP: Evaluación final (evaluate, 50 lotes de 2048)
-----------------------------------------------------------------
[100.0%] Lote 50/50 |    0.4 lotes/s | ETA 0.0s     
Evaluación final terminado en 2m 18.24s

=================================================================
STEP: Evaluación vs Rusher (aggression=0.0) (evaluate, 50 lotes de 2048)
-----------------------------------------------------------------
[100.0%] Lote 50/50 |    0.4 lotes/s | ETA 0.0s     
Evaluación vs Rusher (aggression=0.0) terminado en 2m 10.96s

=================================================================
STEP: Evaluación vs Rusher (aggression=0.5) (evaluate, 50 lotes de 2048)
-----------------------------------------------------------------
[100.0%] Lote 50/50 |    0.4 lotes/s | ETA 0.0s     
Evaluación vs Rusher (aggression=0.5) terminado en 2m 6.05s

=================================================================
STEP: Evaluación vs Rusher (aggression=1.0) (evaluate, 50 lotes de 2048)
-----------------------------------------------------------------
[100.0%] Lote 50/50 |    0.4 lotes/s | ETA 0.0s     
Evaluación vs Rusher (aggression=1.0) terminado en 2m 5.79s
=================================================================
                         FINALIZADO
=================================================================
Perfíl:
========== CUMULATIVE ==========
Wed Sep  9 09:51:30 2026    tiempo.prof

         363193888 function calls (345931643 primitive calls) in 2833.757 seconds

   Ordered by: cumulative time
   List reduced from 16456 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   3362/1    0.610    0.000 2834.153 2834.153 {built-in method builtins.exec}
        1    0.005    0.005 2834.153 2834.153 mainV.py:1(<module>)
        1    0.020    0.020 2829.960 2829.960 mainV.py:165(run)
        8    0.002    0.000 2823.783  352.973 mainV.py:178(_run_step)
        8    0.396    0.049 2819.442  352.430 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\trainerV.py:95(_run)
      400    1.218    0.003 2817.648    7.044 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\trainerV.py:225(_run_batch)
      300    6.756    0.023 1600.170    5.334 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\trainerV.py:492(_replay_turn_and_selection)
        1    0.000    0.000 1471.249 1471.249 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\trainerV.py:57(train)
   144000   13.122    0.000 1295.738    0.009 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\playerAIV.py:108(replay_turn)
     8400    1.184    0.000 1120.197    0.133 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\trainerV.py:282(_run_turn)
     8400   64.484    0.008  787.220    0.094 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Environment\vectorizedEnvironment.py:128(turn)
   187200    2.848    0.000  646.672    0.003 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\playerAIV.py:355(_optimize_step)
    50400   34.698    0.001  556.371    0.011 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Environment\resolve_actions.py:20(resolve_action)
        4    0.000    0.000  521.038  130.259 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\trainerV.py:66(evaluate)
5872200/781950   11.823    0.000  408.055    0.001 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\.venv\Lib\site-packages\torch\nn\modules\module.py:1774(_wrapped_call_impl)
5872200/781950   18.273    0.000  405.327    0.001 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\.venv\Lib\site-packages\torch\nn\modules\module.py:1782(_call_impl)
   463500   42.846    0.000  335.245    0.001 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\turnNetwork.py:33(forward)
      300    1.692    0.006  297.673    0.992 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\trainerV.py:507(_remember_and_replay_selection_batch)
    43200    4.990    0.000  290.582    0.007 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\playerAIV.py:68(replay_selection)
   187200    4.121    0.000  285.926    0.002 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\.venv\Lib\site-packages\torch\optim\optimizer.py:509(wrapper)
   187200    2.344    0.000  267.961    0.001 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\.venv\Lib\site-packages\torch\optim\optimizer.py:60(_use_grad)
   187200    2.263    0.000  264.488    0.001 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\.venv\Lib\site-packages\torch\optim\adam.py:214(step)
    50400   17.189    0.000  241.585    0.005 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Environment\resolve_actions.py:112(_resolve_action_movement)
   187200    0.800    0.000  237.260    0.001 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\.venv\Lib\site-packages\torch\optim\optimizer.py:131(maybe_fallback)
   187200    3.990    0.000  236.301    0.001 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\.venv\Lib\site-packages\torch\optim\adam.py:902(adam)
   187200   10.822    0.000  230.716    0.001 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\.venv\Lib\site-packages\torch\optim\adam.py:553(_multi_tensor_adam)
   144000   21.846    0.000  215.699    0.001 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\playerAIV.py:365(_multi_agent_double_dqn_target)
  2842500    7.408    0.000  214.578    0.000 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\.venv\Lib\site-packages\torch\nn\modules\linear.py:130(forward)
     8400    1.548    0.000  199.272    0.024 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\trainerV.py:695(_turn_mixed_opponent)
   187200    0.935    0.000  195.127    0.001 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\.venv\Lib\site-packages\torch\_tensor.py:566(backward)
   187200    3.787    0.000  194.122    0.001 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\.venv\Lib\site-packages\torch\autograd\__init__.py:255(backward)
  6515600  188.121    0.000  188.121    0.000 {built-in method torch.where}
   187200    2.018    0.000  182.809    0.001 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\.venv\Lib\site-packages\torch\autograd\graph.py:966(_engine_run_backward)
   187200  178.995    0.001  178.995    0.001 {method 'run_backward' of 'torch._C._EngineBase' objects}
   144000   14.590    0.000  178.096    0.001 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\replayMemoryAN.py:71(sample)
    31500    7.560    0.000  167.751    0.005 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\playerAIV.py:232(turn)
  2842500  160.542    0.000  160.542    0.000 {built-in method torch._C._nn.linear}
   463500    7.310    0.000  156.438    0.000 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\.venv\Lib\site-packages\torch\nn\modules\container.py:248(forward)
2891028/1559364   10.201    0.000  152.033    0.000 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\.venv\Lib\site-packages\torch\utils\_contextlib.py:120(decorate_context)
   145070   95.841    0.001  127.586    0.001 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Environment\resolve_actions.py:348(_check_if_targeted)



========== TOTTIME ==========
Wed Sep  9 09:51:30 2026    tiempo.prof

         363193888 function calls (345931643 primitive calls) in 2833.757 seconds

   Ordered by: internal time
   List reduced from 16456 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
  6515600  188.121    0.000  188.121    0.000 {built-in method torch.where}
   187200  178.995    0.001  178.995    0.001 {method 'run_backward' of 'torch._C._EngineBase' objects}
  2842500  160.542    0.000  160.542    0.000 {built-in method torch._C._nn.linear}
   187200   96.863    0.001  100.846    0.001 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\sumTree.py:71(get_batch)
   145070   95.841    0.001  127.586    0.001 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Environment\resolve_actions.py:348(_check_if_targeted)
   187200   90.469    0.000   91.005    0.000 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\replayStorage.py:66(get_batch)
  3154500   81.448    0.000   81.448    0.000 {method 'at' of 'numpy.ufunc' objects}
  2487960   74.135    0.000   74.135    0.000 {method 'gather' of 'torch._C.TensorBase' objects}
    50400   73.840    0.001  105.992    0.002 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Environment\resolve_actions.py:235(_resolve_action_attack)
     8400   64.484    0.008  787.220    0.094 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Environment\vectorizedEnvironment.py:128(turn)
  1761835   57.827    0.000   57.827    0.000 {method 'sum' of 'torch._C.TensorBase' objects}
  3470348   47.690    0.000   47.690    0.000 {method 'float' of 'torch._C.TensorBase' objects}
    43050   46.342    0.001   66.065    0.002 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\playerAIV.py:261(compute_action_mask)
   463500   42.846    0.000  335.245    0.001 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\turnNetwork.py:33(forward)
   194400   42.042    0.000  125.614    0.001 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\sumTree.py:16(_propagate_batch)
   187200   41.293    0.000   41.293    0.000 {built-in method torch._foreach_lerp_}
  1209600   38.139    0.000   38.139    0.000 {method 'scatter_' of 'torch._C.TensorBase' objects}
    50400   34.698    0.001  556.371    0.011 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Environment\resolve_actions.py:20(resolve_action)
   374400   34.475    0.000   34.475    0.000 {built-in method torch._foreach_mul_}
   187200   30.916    0.000   30.916    0.000 {built-in method torch._foreach_addcdiv_}
   374400   29.756    0.000   29.756    0.000 {built-in method torch._foreach_add_}
    13650   28.723    0.002   53.116    0.004 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\player_rusher.py:142(_decidir_turno)
   187200   28.666    0.000   28.666    0.000 {built-in method torch._foreach_div_}
  1784250   28.044    0.000   28.044    0.000 {built-in method torch.relu}
   386978   27.301    0.000   27.301    0.000 {method 'any' of 'torch._C.TensorBase' objects}
  4712570   26.049    0.000   26.049    0.000 {method 'unsqueeze' of 'torch._C.TensorBase' objects}
   128000   23.760    0.000   23.760    0.000 {built-in method torch._C._nn.one_hot}
  1058250   22.854    0.000   24.416    0.000 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\noisy_linear.py:129(weight)
    50400   22.415    0.000   32.823    0.001 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Environment\resolve_actions.py:317(_resolve_action_team_heal)
   144000   21.846    0.000  215.699    0.001 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\playerAIV.py:365(_multi_agent_double_dqn_target)
  3092539   21.744    0.000   21.744    0.000 {method 'view' of 'torch._C.TensorBase' objects}
    18900   20.878    0.001   29.583    0.002 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\player_rusher.py:236(compute_action_mask)
    86250   20.359    0.000   20.359    0.000 {built-in method torch.argmax}
  1155150   19.379    0.000   19.379    0.000 {method 'clone' of 'torch._C.TensorBase' objects}
   187200   18.689    0.000   18.689    0.000 {built-in method torch._foreach_sqrt}
  1058250   18.597    0.000   20.358    0.000 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Agent\noisy_linear.py:137(bias)
5872200/781950   18.273    0.000  405.327    0.001 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\.venv\Lib\site-packages\torch\nn\modules\module.py:1782(_call_impl)
  1403812   18.155    0.000   18.155    0.000 {method 'clamp' of 'torch._C.TensorBase' objects}
    50400   17.259    0.000   22.900    0.000 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Environment\statsV.py:227(accumulate_attacks)
    50400   17.189    0.000  241.585    0.005 C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame\AI\Environment\resolve_actions.py:112(_resolve_action_movement)
### Código del Proyecto:
Todo el código fuente del proyecto (agentes, entorno, redes y configuración) se encuentra subido en los **archivos de contexto de este Proyecto de Claude**. Por favor, léelos directamente desde el almacenamiento del proyecto para realizar las Tareas 2, 3 y 4.

---

## 5. FORMATO Y ESTRUCTURA DE LA RESPUESTA ESPERADA

Quiero que estructures tu respuesta utilizando estrictamente los siguientes títulos y formatos para facilitar mi lectura:

### 📊 BLOQUE 1: DIAGNÓSTICO DE APRENDIZAJE Y ESTRATEGIA (TAREA 1)
*   **Análisis del Estancamiento:** Explica en un párrafo técnico por qué el winrate contra perfiles agresivos escala tan lento con la arquitectura actual.
*   **Lista de Ideas Actualizada:** Muestra la lista del 6 al 12.
*   **Nuevas Propuestas Antiestancamiento:** Añade al menos 3 o 4 propuestas avanzadas nuevas explicando el concepto y por qué frena el estilo "Rusher".

### ⚡ BLOQUE 2: PLAN DE OPTIMIZACIÓN DE RENDIMIENTO (TAREA 2)
*   **Cuellos de Botella Detectados:** Identifica los top 3 problemas de velocidad según el perfilador.
*   **Acciones de Código y PyTorch:** Lista en viñetas las optimizaciones exactas (código, parámetros, configuración de entorno).
*   **Métrica Objetivo:** Confirma si con esto es viable bajar de 10 minutos los 100 episodios sin cambiar el tamaño de replay.

### 🧹 BLOQUE 3: AUDITORÍA DE ARQUITECTURA Y CLEAN CODE (TAREA 3)
*   **Diagnóstico de Clases:** Una tabla corta o lista de qué clases están violando principios SOLID o acumulando demasiadas responsabilidades.
*   **Plan de Acción Paso a Paso:** Un mapa de ruta secuencial (Paso 1, Paso 2...) para refactorizar sin romper la lógica de RL.

### 🧠 BLOQUE 4: NUEVAS MECÁNICAS DE RL (TAREA 4)
Presenta tus recomendaciones usando estrictamente una tabla con las siguientes columnas para comparar opciones:

| Mecánica Recomendada | Impacto en Aprendizaje (Alto/Medio) | Coste en Tiempo de Ejecución (Alto/Medio/Bajo) | Coste de Implementación (Complejo/Medio/Fácil) | Justificación Breve |
| :--- | :--- | :--- | :--- | :--- |

Solo quiero que hagas una evaluación del estado del proyecto, el archivo es un resumen de cosas que se han dejado por hacer. Pero no es extremadamente relevante. En el estado del proyecto quiero que seas realmente honesto respecto a:
-Limpieza del código y responabilidades individuales de las clases
-Capacidad de aprendizaje de la IA, entorno de entrenamiento adecuado, pesos de las recompensas bien calculadas.
-Cosas a implementar ya sean a nivel de aprendizaje o a nivel de mecánicas en el videojuego
-Posibles optimizaciones a nivel de eficiencia temporal y de entrenamiento
-Configuraciones del entrono que no este realizando
-Cosas importantes de deep learning (pytorch) que no este realizando
-Cosas que tengo qeu implementar (incluidas las del resumen) y lista de prioridades.
Quiero honestidad, no asumas que mi código es perfecto, busca cualquier tipo de error, cosa mejorable, cosa inutil, código muerto, y dimelo. 