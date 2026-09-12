# Registro de constantes

Última revisión: 2026-09-12

Este documento cataloga las constantes de `constants.py` que **merecen explicación**
(valor, hipótesis, efecto esperado, resultado empírico). Las constantes triviales
(contadores estructurales) y las que no están en uso se listan al final.

---

## Meta-juego (Castillo)

### USE_META_GAME
- Valor actual: `True`
- Hipótesis: modo de draft desde castillo (10 slots) vs catálogo de 5 tipos fijos
- Efecto esperado: `False` → draft clásico con 5 tipos; `True` → draft desde castillo con instancias únicas
- Resultado empírico: n/a
- Estado: **Activa, no se ha experimentado cambiarla**

### MAX_BATALLAS
- Valor actual: `10`
- Hipótesis: edad máxima de un héroe antes de morir por envejecimiento
- Efecto esperado: mayor → héroes duran más en el castillo, menos presión de renovación
- Resultado empírico: -
- Estado: sin validar

### GOLD_INICIAL
- Valor actual: `250`
- Hipótesis: oro de partida para comprar héroes
- Efecto esperado: mayor → castillo más poblado desde el inicio; menor → más restricción de recursos
- Resultado empírico: -
- Estado: sin validar

### COST_COMPRA
- Valor actual: `30`
- Hipótesis: coste de añadir un héroe al castillo tras una muerte
- Efecto esperado: `GOLD_POR_BATALLA / COST_COMPRA` determina cuántos héroes se pueden reponer por batalla
- Resultado empírico: -
- Estado: sin validar

### GOLD_POR_BATALLA
- Valor actual: `100`
- Hipótesis: ingreso fijo de oro tras cada batalla
- Efecto esperado: alto → economía holgada; bajo → economía restrictiva
- Resultado empírico: -
- Estado: sin validar

### GOLD_NORM_REF
- Valor actual: `500`
- Hipótesis: referencia para normalizar el oro en la observación del draft
- Efecto esperado: división `oro / GOLD_NORM_REF` produce valores típicamente en `[0, 1.5]`
- Resultado empírico: -
- Estado: sin validar. **Ojo:** el rango natural del oro es `[GOLD_INICIAL, GOLD_INICIAL + N·GOLD_POR_BATALLA]`, muy por encima de 500. Si el valor medio ronda 300-500, la normalización centra la señal; si supera 1000, satura.

### WARRIOR_USE_EMA_DECAY
- Valor actual: `0.995`
- Hipótesis: decaimiento de la media móvil de uso reciente por tipo de guerrero
- Efecto esperado: mayor → memoria larga, estrategia de compra más estable; menor → reacción rápida al meta actual
- Resultado empírico: -
- Estado: sin validar

### SHOP_TEMPERATURE
- Valor actual: `0.2`
- Hipótesis: temperatura del softmax sobre la EMA de uso para muestrear el tipo de héroe comprado
- Efecto esperado: menor → más determinista (favorece el tipo más usado); mayor → más diversidad
- Resultado empírico: -
- Estado: sin validar

### TURN_STATE_DIM
- Valor actual: `58 + 24 * MAX_POOL_SIZE + 12 + 4` = 218
- Hipótesis: dimensión del vector de observación de TurnNetwork
- Efecto esperado: debe coincidir exactamente con lo que produce `ObservationV.normalize_batch`; +4 corresponde al perfil del rival
- Resultado empírico: n/a
- Estado: **Activa, cálculo derivado de otras constantes**

---

## IA — Exploración (epsilon-greedy)

### EPSILON_SELECTION
- Valor actual: `0.5`
- Hipótesis: epsilon inicial de exploración para `SelectionNetwork`
- Efecto esperado: alto → draft más aleatorio al principio; bajo → dependencia temprana de la red
- Resultado empírico: -
- Estado: sin validar

### EPSILON_SEL_MIN
- Valor actual: `0.05`
- Hipótesis: suelo del epsilon de selección durante el decay
- Efecto esperado: mantener siempre algo de exploración en el draft
- Resultado empírico: -
- Estado: sin validar

### EPSILON_SEL_DECAY
- Valor actual: `0.99`
- Hipótesis: decaimiento por lote del epsilon de selección
- Efecto esperado: tras 100 lotes, epsilon ≈ 0.5 × 0.99^100 ≈ 0.18 → ya en el suelo efectivo
- Resultado empírico: -
- Estado: sin validar

### EPSILON_RESIDUAL
- Valor actual: `0.01`
- Hipótesis: epsilon residual aplicado siempre, incluso en entrenamiento
- Efecto esperado: garantiza un mínimo de acciones no-greedy
- Resultado empírico: -
- Estado: sin validar

---

## IA — Aprendizaje

### SELECTION_LEARNING_RATE
- Valor actual: `0.0001`
- Hipótesis: lr de Adam para `SelectionNetwork`
- Efecto esperado: mayor → aprendizaje más rápido con más varianza; menor → más estable pero más lento
- Resultado empírico: -
- Estado: sin validar

### TURN_LEARNING_RATE
- Valor actual: `0.0001`
- Hipótesis: lr de Adam para `TurnNetwork`
- Efecto esperado: idéntico al anterior, pero para la red de combate
- Resultado empírico: -
- Estado: sin validar

### SELECTION_REPLAY_DATA
- Valor actual: `131_072`
- Hipótesis: capacidad del buffer de selección
- Efecto esperado: mayor → más memoria de drafts previos; menor → sobreajuste al draft reciente
- Resultado empírico: - (a ~6144 nuevas exp/lote, rota completo en ~21 lotes)
- Estado: sin validar. **Candidato a retest con más lotes** si el patrón de buffer de turno se confirma.

### TURN_REPLAY_DATA
- Valor actual: `262_144`
- Hipótesis: capacidad del buffer de turno
- Efecto esperado: mayor → mejor diversidad, menos olvido catastrófico; pero tarda más en llenarse
- Resultado empírico: 262k a 50 lotes → self-play +6.4pp con signos inconsistentes en rusher; no concluyente
- Estado: **No conclusivo a 50 lotes** (buffer tarda ~13 lotes en llenarse). Pendiente retest a 100+ lotes.

### BATCH_SIZE
- Valor actual: `256`
- Hipótesis: tamaño del batch de replay
- Efecto esperado: mayor → gradiente más suave, pero más cómputo por update; menor → más varianza
- Resultado empírico: se probó 128 en un run (con COPY_DQN alto); degradó el winrate
- Estado: sin validar en aislamiento

### DISCOUNT_FACTOR
- Valor actual: `0.95`
- Hipótesis: gamma para los retornos
- Efecto esperado: horizonte efectivo ≈ 1/(1-γ) = 20 turnos, cubre la partida completa
- Resultado empírico: -
- Estado: sin validar

### GRAD_CLIP_MAX_NORM
- Valor actual: `1.0`
- Hipótesis: clip global de gradiente
- Efecto esperado: previene explosiones de gradiente en batches con TD errors extremos (típicos de PER)
- Resultado empírico: -
- Estado: sin validar

### COPY_DQN_SEL
- Valor actual: `120`
- Hipótesis: cada cuántos replays se sincroniza la target network de `SelectionNetwork`
- Efecto esperado: mayor → target más estable; menor → target sigue la online más rápido
- Resultado empírico: pendiente — con `SELECTION_REPLAYS_PER_BATCH=72`, 120 significa ~0.6 syncs/lote
- Estado: **Adoptado recientemente**, pendiente confirmación con más seeds

### COPY_DQN_TURN
- Valor actual: `150`
- Hipótesis: cada cuántos replays se sincroniza la target network de `TurnNetwork`
- Efecto esperado: mayor → bootstrapping más estable; menor → target sigue la online más rápido
- Resultado empírico: 150 → +12.25 pp winrate a 200 lotes (vs 50)
- Estado: **Validado parcialmente** — falta replicar con más seeds y con el resto de fases confirmadas

---

## Juego — Recompensas

### REWARD_WEIGHTS
- Valor actual: ver diccionario
- Hipótesis: pesos individuales de cada término del shaping
- Efecto esperado por término:
  - `damage = 2` — daño neto (P1 − P2); dominante
  - `deaths = 20` — muertes causadas; multiplica el impacto de un kill
  - `win = 1` — multiplicador de `WIN_REWARD`
  - `blocks = 1` — daño bloqueado; señal continua
  - `heal = 1` — curación efectiva
  - `shaping_weight = 10` — término de diferencia de vida descontada
  - `wasted_heal = -5` — castigo por curación sin efecto
  - `wasted_defense = -5` — castigo base por defensa sin bloqueo
  - `strategic_movement = 5` — bonus base por reposicionamiento útil
  - `overkill_damage = -5` — castigo por daño excedido
  - `kill_confirmed = 0.5` — bonus por remate efectivo
- Resultado empírico: -
- Estado: **Algunos términos son candidatos a ablation** (`overkill_damage`, `wasted_defense`)

### TURN_PENALTY_BASE
- Valor actual: `2`
- Hipótesis: penalización de turno durante la fase temprana
- Efecto esperado: valor bajo, no debe empujar al agente a terminar rápido antes de tiempo
- Resultado empírico: -
- Estado: sin validar

### TURN_PENALTY_RAMP_START
- Valor actual: `4.5`
- Hipótesis: turno a partir del cual la penalización empieza a crecer linealmente
- Efecto esperado: subirlo → partidas más largas sin castigo; bajarlo → presión temprana
- Resultado empírico: -
- Estado: sin validar

### TURN_PENALTY_RAMP_TURNS
- Valor actual: `12`
- Hipótesis: turnos que tarda la penalización en pasar de `BASE` a `MAX`
- Efecto esperado: subirlo → rampa más suave; bajarlo → castigo tardío más agresivo
- Resultado empírico: -
- Estado: sin validar

### TURN_PENALTY_MAX
- Valor actual: `35`
- Hipótesis: techo de la penalización cerca del límite de turnos
- Efecto esperado: con `REWARD_SCALE=100`, es 0.35 por turno tardío; acumulado ≈ 3.7/partida (vs win reward = 10 normalizado)
- Resultado empírico: reducirlo a 25 → **−13.5 pp self-play y varianza entre seeds explotó**
- Estado: **Cerrado, mantener en 35**

### WIN_REWARD
- Valor actual: `1000`
- Hipótesis: recompensa por ganar
- Efecto esperado: referencia de magnitud para el resto del shaping
- Resultado empírico: -
- Estado: sin validar

### DRAW_PENALTY
- Valor actual: `200` (20 % de `WIN_REWARD`)
- Hipótesis: castigo por empate, ligeramente menor que perder para no incentivar tablas
- Efecto esperado: subirlo → agente arriesga más en finales igualados; bajarlo → prefiere asegurar empate
- Resultado empírico: -
- Estado: sin validar

### REWARD_SCALE
- Valor actual: `100.0`
- Hipótesis: divisor de normalización del reward final
- Efecto esperado: `WIN_REWARD / REWARD_SCALE = 10` es la escala de referencia del retorno por partida
- Resultado empírico: -
- Estado: sin validar

### MAX_TURNS
- Valor actual: `20`
- Hipótesis: límite de turnos por partida
- Efecto esperado: condiciona la magnitud de `TURN_PENALTY_MAX` (ver arriba)
- Resultado empírico: `avg_turns` actual ≈ 7-10, holgura amplia hasta el límite
- Estado: sin validar

### MAX_DEATHS_PER_TEAM
- Valor actual: `3`
- Hipótesis: bajas por bando antes de terminar la partida
- Efecto esperado: 3 = todos los guerreros; reduce a 1-2 → partidas más cortas
- Resultado empírico: -
- Estado: sin validar

---

## Reward shaping condicional al perfil del rival

### WASTED_DEFENSE_WEIGHT_MAX_AGGRO
- Valor actual: `REWARD_WEIGHTS["wasted_defense"]` (−5). **Nota:** el comentario del código dice "-1.5 (vs -5 en REWARD_WEIGHTS)" pero el valor real asignado es −5. Es el valor de control degenerado.
- Hipótesis: a agresividad=1 (rival agresivo), penalizar menos la defensa sin bloqueo; a agresividad=0, mantener −5
- Efecto esperado: valor > −5 → incentivo a defender contra rivales agresivos
- Resultado empírico: -
- Estado: **Control degenerado activo** (interpolación colapsa al comportamiento previo). Activar a −1.5 para el test real.

### STRATEGIC_MOVEMENT_WEIGHT_MAX_AGGRO
- Valor actual: `REWARD_WEIGHTS["strategic_movement"]` (+5). Comentario dice "+9" pero valor real +5. Control degenerado.
- Hipótesis: a agresividad=1, premiar más el reposicionamiento defensivo; a agresividad=0, mantener +5
- Efecto esperado: valor > +5 → incentivo a reposicionarse contra rivales agresivos
- Resultado empírico: -
- Estado: **Control degenerado activo.** Activar a +9 para el test real.

---

## Pool de oponentes

### MAX_MODELS
- Valor actual: `50`
- Hipótesis: tamaño máximo de la pool de snapshots
- Efecto esperado: mayor → pool más diversa; menor → pool más reciente
- Resultado empírico: -
- Estado: sin validar

### SAVE_MODEL_FRACTION
- Valor actual: `0.05`
- Hipótesis: cada cuánto (fracción del total) guardar un snapshot
- Efecto esperado: con 200 lotes, guarda uno cada 10
- Resultado empírico: -
- Estado: sin validar

### POOL_RANGE_FRACTION
- Valor actual: `0.01`
- Hipótesis: cada cuánto refrescar la asignación de oponentes desde pool
- Efecto esperado: con 200 lotes, refresca cada 2
- Resultado empírico: -
- Estado: sin validar

### POOL_PORCENTAGE
- Valor actual: `0.3`
- Hipótesis: fracción de partidas del lote que usan un oponente de la pool
- Efecto esperado: mayor → más exposición a políticas antiguas; menor → más self-play puro
- Resultado empírico: -
- Estado: sin validar

---

## Replay y priorización (PER)

### SELECTION_REPLAYS_PER_BATCH
- Valor actual: `72`
- Hipótesis: número de updates de `SelectionNetwork` por lote
- Efecto esperado: mayor → más aprendizaje por lote; combinado con `COPY_DQN_SEL`, determina syncs/lote
- Resultado empírico: -
- Estado: sin validar

### TURN_REPLAYS_PER_BATCH
- Valor actual: `240`
- Hipótesis: número de updates de `TurnNetwork` por lote
- Efecto esperado: idéntico, para Turn; con `COPY_DQN_TURN=150` → ~1.6 syncs/lote
- Resultado empírico: -
- Estado: sin validar

### ALPHA
- Valor actual: `0.8`
- Hipótesis: exponente de priorización del PER
- Efecto esperado: mayor → muestreo muy sesgado hacia transiciones con TD error alto; menor → más uniforme
- Resultado empírico: -
- Estado: sin validar. **Candidato a ablation** si se sospecha lock-in de prioridades.

### BETA_START
- Valor actual: `0.4`
- Hipótesis: exponente inicial del weighted importance sampling
- Efecto esperado: valores bajos al principio corrigen menos el sesgo de PER
- Resultado empírico: -
- Estado: sin validar

### BETA_END
- Valor actual: `1.0`
- Hipótesis: exponente final (β=1 → corrección completa del sesgo)
- Efecto esperado: `1.0` es el valor estándar de la literatura
- Resultado empírico: -
- Estado: sin validar

### BETA_DECAY_RATE
- Valor actual: `0.9999`
- Hipótesis: ritmo de transición de β entre `BETA_START` y `BETA_END` por replay
- Efecto esperado: con TURN_REPLAYS_PER_BATCH=240, tarda ~6700 replays en llegar a β≈1.0
- Resultado empírico: -
- Estado: sin validar. **Posible calibración mal ajustada** para 50-100 lotes (β apenas se mueve de 0.4).

### PER_EPSILON
- Valor actual: `0.1`
- Hipótesis: pequeño epsilon para evitar prioridades cero
- Efecto esperado: estabilidad numérica, sin efecto semántico significativo
- Resultado empírico: -
- Estado: sin validar

---

## N-step y arquitectura

### N_STEP
- Valor actual: `3`
- Hipótesis: número de pasos para los retornos n-step
- Efecto esperado: mayor → propagación de recompensa más rápida, target menos sesgada por bootstrap; menor → comportamiento estándar
- Resultado empírico: -
- Estado: sin validar

### USE_DUELING_DQN
- Valor actual: `True`
- Hipótesis: dueling architecture en TurnNetwork
- Efecto esperado: `True` → separa value y advantage; mejora aprendizaje en estados con muchas acciones indiferentes
- Resultado empírico: -
- Estado: sin validar

### NOISY_SIGMA_INIT
- Valor actual: `0.5`
- Hipótesis: sigma inicial de las capas `NoisyLinear`
- Efecto esperado: mayor → más exploración al inicio; menor → política más determinista desde el principio
- Resultado empírico: -
- Estado: sin validar

### SIGMA_MIN
- Valor actual: `0.05`
- Hipótesis: suelo de `weight_sigma`/`bias_sigma` tras cada update
- Efecto esperado: previene colapso de exploración durante el entrenamiento
- Resultado empírico: -
- Estado: sin validar. **Candidato a revisión** si se observa que sigma colapsa en runs largos.

### RESET_IN_DECISIONS
- Valor actual: `True`
- Hipótesis: reset de ruido NoisyNet por decisión (paper original) vs por lote
- Efecto esperado: `True` → más diversidad intra-partida; `False` → ruido constante durante toda la partida
- Resultado empírico: `True` → +3.5 pp vs rusher 0.0/0.5/1.0 sin coste en self-play
- Estado: **Adoptado provisionalmente**, pendiente confirmación con más seeds

### USE_TORCH_COMPILE
- Valor actual: `False`
- Hipótesis: compilación JIT de las redes
- Efecto esperado: puede acelerar el forward/backward; complejidad en el manejo de `_orig_mod` al hacer `load_state_dict`
- Resultado empírico: -
- Estado: sin validar

### DELETE_DIRECTORIES
- Valor actual: `True`
- Hipótesis: limpiar directorios de modelos al iniciar el entrenamiento
- Efecto esperado: `True` en self-play, **peligroso en evaluación** — puede borrar checkpoints que quieres cargar
- Resultado empírico: -
- Estado: activo, revisar contexto

---

## ELO

### ELO_INITIAL
- Valor actual: `1000.0`
- Hipótesis: rating inicial de P1, P2 y cualquier snapshot nuevo
- Efecto esperado: referencia del sistema de matchmaking
- Resultado empírico: -
- Estado: sin validar

### ESTANDAR_ELO
- Valor actual: `150`
- Hipótesis: divisor del argumento de la exponencial en `expected_score`
- Efecto esperado: mayor → más inercia del Elo (partidas cuentan menos); menor → Elo más volátil
- Resultado empírico: -
- Estado: sin validar

### K_FACTOR_ELO
- Valor actual: `32`
- Hipótesis: multiplicador del delta de Elo por partida
- Efecto esperado: `32` es el estándar ajedrecístico; valores mayores → oscilaciones rápidas
- Resultado empírico: -
- Estado: sin validar

### ELO_TEMPERATURE
- Valor actual: `15`
- Hipótesis: temperatura del softmax en el muestreo de oponentes de pool por cercanía de Elo
- Efecto esperado: menor → oponentes muy cercanos al agente; mayor → pool más diversa
- Resultado empírico: -
- Estado: sin validar

---

## Orquestación (YAML)

### N_BATCH
- Valor actual: `2048`
- Hipótesis: número de partidas paralelas por lote
- Efecto esperado: determina el throughput y el tamaño de las evaluaciones
- Resultado empírico: -
- Estado: sin validar

### TRAIN_EPISODES / EVAL_EPISODES
- Valor actual: `200` / `20`
- Hipótesis: número de lotes de entrenamiento y de evaluación
- Efecto esperado: más entrenamiento → mejor política hasta saturar; más evaluación → menor varianza de las métricas
- Resultado empírico: 200 lotes mostró colapso en seed 42 (ver análisis). 100 lotes parece el punto dulce actual.
- Estado: **100 lotes es el valor recomendado** por ahora

### SEED
- Valor actual: `None` (ignorado con `MULTI_SEED_ENABLED=True`)
- Hipótesis: semilla de reproducibilidad
- Efecto esperado: `42`, `43`, `44` son las habituales
- Resultado empírico: seed 43 suele desbalancear self-play (Elo P1−P2 grande). Documentar como caso a vigilar.
- Estado: activa

### USE_WANDB / USE_GUI / RUN_* (flags de orquestación)
- Valor actual: `USE_WANDB=False`, `USE_GUI=True`, `RUN_SELF_PLAY=True`, `RUN_EVALUATION=True`, `RUN_RUSHER_TESTS=True`, `RUN_RUSHER_FINETUNE=False`
- Hipótesis: flags que controlan el pipeline
- Efecto esperado: autodescriptivos
- Estado: activas, sin ablations individuales

### RUSHER_OPPONENT_PERCENTAGE
- Valor actual: `0.3` (pero `MULTI_SEED_BASELINE` y config actual lo ponen a `0.0`)
- Hipótesis: fracción de partidas contra rusher durante self-play
- Efecto esperado: > 0 → exposición al rusher en entrenamiento
- Resultado empírico: **0.15 y 0.30 degradan el winrate global**. Confirmado.
- Estado: **Cerrado, mantener 0.0**

### RUSHER_TEST_EPISODES
- Valor actual: `20`
- Hipótesis: número de lotes de evaluación vs rusher
- Efecto esperado: con N=2048, 20 lotes = 40 960 partidas por banda; precisión ±0.2 pp
- Estado: activa

### RUSHER_AGGRESSION_BANDS / _BAND_WEIGHTS
- Valor actual: `[(0.0,0.3),(0.35,0.65),(0.7,1.0)]` con pesos `[0.25,0.35,0.40]`
- Hipótesis: muestreo por bandas de agresividad del rusher durante entrenamiento (solo si `RUSHER_OPPONENT_PERCENTAGE>0`)
- Efecto esperado: más exposición a la banda donde la IA peor rinde
- Resultado empírico: no aplicable — rusher no entra en self-play
- Estado: **inactivo en el pipeline actual**

### RUSHER_FINETUNE_PHASES
- Valor actual: `[(0.15,0.0,0.3),(0.35,0.3,0.6),(0.5,0.6,1)]`
- Hipótesis: división de episodios de FT por fases de agresividad
- Efecto esperado: FT progresivo desde pasivo a agresivo
- Resultado empírico: **el FT dedicado degrada el winrate global**, incluso contra el propio rusher
- Estado: **cerrado, desactivado**

### PROFILE_EMA_DECAY
- Valor actual: `0.7`
- Hipótesis: ritmo de actualización de la EMA del perfil del rival
- Efecto esperado: mayor → perfil más pegajoso, más suave; menor → más reactivo a cambios recientes
- Resultado empírico: -
- Estado: sin validar. **Relevante para el nuevo reward shaping condicional** — si la agresividad se queda en 0.4-0.6, el cambio no se manifiesta.

### PROFILE_DAMAGE_POTENTIAL_REF
- Valor actual: `75` (Rogue PoisonGas 7×3 + Wizard Fireball 7×3 + Cleric HolyNova 9×3)
- Hipótesis: referencia para normalizar el prior de agresividad en `set_initial_priors`
- Efecto esperado: denominador del cálculo `total_damage / PROFILE_DAMAGE_POTENTIAL_REF`
- Estado: activa

---

## Multi-seed

### MULTI_SEED_ENABLED
- Valor actual: `False` (config.yaml lo sobreescribe a `True`)
- Hipótesis: activa la orquestación de varios runs con seeds distintas
- Efecto esperado: `True` → ignora `SEED` y usa `MULTI_SEED_LIST`; `False` → una sola seed
- Estado: activa

### MULTI_SEED_LIST
- Valor actual: `[42, 43, 44]`
- Hipótesis: lista de seeds a ejecutar secuencialmente
- Estado: activa

### MULTI_SEED_BASELINE
- Valor actual: `None`
- Hipótesis: ruta(s) al baseline para la comparación final
- Efecto esperado: acepta `None`, ruta única, lista de rutas, o carpeta contenedora con subcarpetas `sN/`
- Estado: activa

---

# Constantes no usadas o de bajo interés

Estas no tienen entrada propia porque o no se leen en el pipeline actual, o su significado es auto-evidente.

## Sin uso en el pipeline actual (fantasma)

Se mantienen por compatibilidad o como restos de iteraciones previas. **Candidatas a eliminar** en la próxima limpieza de `constants.py`.

- `EPSILON_TURN` — `AIPolicy.turn()` no la lee. La exploración de Turn corre a cargo exclusivamente de NoisyNet.
- `EPSILON_TURN_MIN` — idem.
- `EPSILON_TURN_DECAY` — idem.
- `COPY_DQN` — reemplazado por `COPY_DQN_SEL` y `COPY_DQN_TURN`. Ya no se usa en `AgentTrainer`.

## Triviales (sin hipótesis que documentar)

Contadores y etiquetas estructurales. Cambiarlos invalida checkpoints pero no son objeto de ablations.

- `WARRIOR_QUANTITY = 5`
- `MAX_POOL_SIZE = 6`
- `ABILITIES_PER_WARRIOR = 4`
- `NUM_SLOTS = 3`
- `MAX_CASTLE_SIZE = 10`
- `MAX_ABILITY_LEVEL = 5`
- `VERSION`
- `RUN_NAME_SUFFIX`
- `PROFILE_CPROFILE_OUTPUT`
- `PROFILE_TORCH`, `PROFILE_TORCH_BATCHES`, `PROFILE_CPROFILE`
- `RUSHER_TEST_EPSILON` — siempre 0.0 en la práctica
- `PLAY_EPSILON` — siempre 0.0 en la práctica
- `HUMAN_*` — no se usa salvo modo humano explícito
- `PLAY_*`, `PLAY_AGAINST_AI` — modo manual, poco frecuente