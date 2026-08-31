1. Visión General
Género: Estrategia por turnos (1v1) con gestión de recursos y meta‑juego.
Batalla (micro):
Dos jugadores (P1 y P2) eligen un equipo de 3 héroes de entre su colección (castillo) y se enfrentan en un tablero de 3 posiciones (front, mid, back). Cada héroe tiene 4 habilidades activas de una pool de 6 (sorteo por instancia al comprar). Las acciones posibles son: 4 habilidades (ataque, cura, defensa) + 2 movimientos (cambiar de posición). El orden de turnos se decide por velocidad. La partida dura entre 10 y 30 turnos (objetivo de diseño).
Meta‑juego (macro):
Después de cada batalla se obtienen recompensas (oro) que se invierten en el castillo para:
Comprar nuevos héroes (con habilidades sorteadas aleatoriamente).
Gestionar el envejecimiento (mueren tras un número de batallas).
(Futuro) Mejorar habilidades y estadísticas.
IA:
Agente DQN con dos redes (selección de equipo y acciones de turno). Entrenamiento en self‑play con una pool de oponentes (snapshots de P2) y matchmaking por Elo para un currículum automático.

2. Arquitectura Técnica (Estado Actual)
Entorno vectorizado:
VectorizedEnvironment ejecuta N=2048 partidas en paralelo usando tensores de PyTorch. Todas las operaciones están vectorizadas (movimiento, ataques, cura, defensa, cooldowns, etc.).
Agente: PlayerAIV con dos redes:
SelectionNetwork: elige 3 héroes del castillo (acción = slot_id * 3 + posición).
Capas: Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(32) -> NoisyLinear(salida).
TurnNetwork: elige una acción por cada héroe vivo (6 acciones: 4 habilidades + 2 movimientos).
Arquitectura Dueling DQN (valor de estado + ventaja por slot).
Capas compartidas Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(32), luego NoisyLinear para value y advantage.
Algoritmo de RL:
Doble DQN (red online + target, copia cada 50 pasos de replay).
Dueling DQN (ya implementado, activo por defecto).
Prioritized Experience Replay (PER) con SumTree vectorizado y almacenamiento en ReplayStorage (estructura SoA).
N‑step returns (actualmente N_STEP=3, ajustable).
Noisy Networks (sustituye a ε‑greedy; las redes tienen ruido parametrizado que se resetea en cada decisión y en cada replay).
Matchmaking por Elo: la pool de oponentes asigna checkpoints según la distancia de Elo (softmax con temperatura), y los Elos se actualizan tras cada lote.
Pool de oponentes: OpponentPoolV guarda snapshots de P2 (modelos completos) y los usa para enfrentarse a P1 en un porcentaje de partidas (POOL_PORCENTAGE=0.3).
Meta‑juego: CastleV gestiona el castillo (tipos, habilidades, niveles, edad, alive, oro). Al final de cada partida se ejecuta _run_meta_step(): envejecimiento, muerte en combate, compra de nuevos héroes con heurística (shop_heuristics).
Optimizaciones:
Tiempo de run reducido de ~600s a ~120s para 5 lotes con N=2048 (≈80% mejora) gracias a vectorización de mask_turn, SumTree, y reestructuración del buffer.

3. Decisiones de Diseño (Tomadas y Pendientes)
Aspecto
Decisión / Estado
Duración de partidas
Objetivo 10‑30 turnos. Se han ajustado recompensas: WIN_REWARD=1500, TURN_PENALTY_BASE=8, shaping_weight=2. En pruebas: la duración media actual es ~7.6 turnos (según la última ejecución), por lo que se necesita reajustar (bajar WIN_REWARD o aumentar shaping_weight).
Habilidades por instancia
✅ Implementado (Opción B): cada héroe comprado tiene su propio sorteo de 4 habilidades (de una pool de 6) que permanece fijo toda su vida. Las habilidades activas se guardan en p1_instance_abilities / p2_instance_abilities y se incluyen en la observación (one‑hot).
Pasivas comunes
❌ Pendiente. Serán habilidades compartidas por todos los héroes, sorteadas por partida. Se aplazan.
Meta‑juego (Castillo)
✅ Implementado (heurístico): compra automática con heurística find_first_missing_type (rellena tipos faltantes) + envejecimiento (MAX_BATALLAS=10) + muerte en combate. La compra usa GOLD_INICIAL=250, COST_COMPRA=40, GOLD_POR_BATALLA=100. Límite de 10 héroes por castillo.
Mejoras de habilidades
❌ Pendiente. Se hará con heurística (no RL por ahora).
Perfil del rival
❌ Pendiente. Se añadirá un vector de contexto con estadísticas de enfrentamientos (heurístico).
Unity / Exportación
❌ Pendiente. Se usará ONNX + Barracuda cuando el juego esté balanceado y el modelo estable (>60% winrate).


4. Estado de las Técnicas de RL (Evaluadas)
Técnica
Implementada
Decisión / Estado
Doble DQN
✅
Imprescindible (ya presente).
Dueling DQN
✅
Imprescindible (ya presente).
PER (Prioritized ER)
✅
Imprescindible (ya presente).
N‑step Returns
✅
N_STEP=3; se probará con 1 o 2 para partidas cortas.
Noisy Networks
✅
Implementada. Sustituye a ε‑greedy; mejora exploración.
Matchmaking por Elo
✅
Implementado. Currículum automático para la pool de oponentes.
PopArt
❌
Aceptada (condicionada a LSTM/memoria de roster). Se aplaza.
Distributional RL (C51)
❌
Rechazada (partidas cortas y deterministas).
LSTM / Memoria intra‑partida
❌
Retrasada (estado Markoviano en 10‑30 turnos, no necesario).
League Training
❌
Rechazada (coste desproporcionado).
Soft target updates
❌
Por decidir (evaluar si mejora estabilidad).
Gradient clipping
❌
Por decidir (red de seguridad barata).


5. Roadmap de Implementación (Actualizado)
Fase 0: Estabilización Inmediata (COMPLETADA)
✅ Corregir bug de movimiento (intercambio de health y cooldowns).
✅ Añadir flush() en NStepBuffer.
✅ Corregir opp_types en TrainerV._run_batch.
✅ Ajustar recompensas y MAX_TURNS a 30.
✅ Corregir bug de cooldowns (reinicio al inicio del turno).
✅ Validar mask_turn vectorizado (pruebas torch.equal).
Fase 1: Mejoras de RL (COMPLETADA)
✅ Implementar Noisy Networks en SelectionNetwork y TurnNetwork.
✅ Implementar Matchmaking por Elo en OpponentPoolV.
✅ Ajustar N_STEP a 3 (pendiente probar 1 o 2).
✅ Decidir sobre Soft target updates y gradient clipping (por ahora no se usan).
Fase 2: Mecánicas de Juego (meta‑juego base) (COMPLETADA)
✅ Sorteo 4‑de‑6 habilidades por instancia (Opción B).
✅ Modificar reset() para sortear habilidades por partida.
✅ Modificar ObservationV.normalize_batch para incluir one‑hot de habilidades activas.
✅ Adaptar mask_turn para usar habilidades de la instancia.
✅ Heurística de compra/mejora (shop_heuristics).
✅ Envejecimiento y muerte de héroes (CastleV).
✅ Límite de 10 guerreros en el castillo.
❌ Pendiente: Perfil del rival (estadísticas simples, vector de contexto en observación).
Fase 3: Entrenamiento y Balanceo (EN CURSO)
🔄 Entrenar la IA con todas las mecánicas activas (ejecución de 5 lotes, 2048 partidas/lote).
🔄 Rebalancear recompensas para lograr duración media de 10‑30 turnos (actualmente ~7.6).
🔄 Ajustar hiperparámetros (learning rate, batch size, replays por lote).
🔄 Validar winrate de P1 vs pool (>60% después de 20 lotes).
Fase 4: Unity y Producto (PENDIENTE)
⬜ Exportar modelo a ONNX.
⬜ Integrar con Unity (Barracuda).
⬜ Pruebas de usuario y fine‑tuning con humanos.
⬜ (Opcional) RL para el meta‑juego si el juego triunfa.

6. Métricas de Éxito (Objetivos)
Métrica
Valor objetivo
Estado actual
Duración media de partida
10‑30 turnos
~7.6 turnos (necesita ajuste).
Winrate de P1 vs pool
> 60%
En evaluación (primera ejecución muestra Elo P1=1040, P2=1016).
Uso de habilidades
≥3 de 4 activas
En evaluación (se ve en estadísticas de ataques).
Diversidad de héroes
Diferentes combinaciones
En evaluación (selección aún por analizar).



Fecha de la actualización: 2026-08-31
Responsable: Ángel Méndez Prieto


