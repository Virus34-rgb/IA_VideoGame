"""
Constantes globales del proyecto Castle Game.

Agrupadas por área: IA, juego, recompensas, pool de oponentes, PER, N-step.
"""

# ============================================================
# IA - Parámetros de exploración (epsilon-greedy)
# ============================================================
EPSILON_SELECTION = 0.5           # Épsilon inicial para selección de equipo
EPSILON_TURN = 0.5                # Épsilon inicial para acciones de turno
EPSILON_SEL_MIN = 0.05            # Épsilon mínimo para selección
EPSILON_TURN_MIN = 0.05           # Épsilon mínimo para turno
EPSILON_SEL_DECAY = 0.99998       # Decaimiento por lote para selección
EPSILON_TURN_DECAY = 0.99998      # Decaimiento por lote para turno

# ============================================================
# IA - Parámetros de aprendizaje
# ============================================================
SELECTION_LEARNING_RATE = 0.0001  # Learning rate para red de selección
TURN_LEARNING_RATE = 0.0001       # Learning rate para red de turno
SELECTION_REPLAY_DATA = 500_000 # Capacidad del buffer de selección
TURN_REPLAY_DATA = 500_000      # Capacidad del buffer de turno
BATCH_SIZE = 64                   # Tamaño del batch de replay
DISCOUNT_FACTOR = 0.95            # Factor de descuento (gamma)
COPY_DQN = 50                     # Frecuencia de copia a target network (en pasos de replay)

# ============================================================
# Juego - Reglas básicas
# ============================================================
WARRIOR_QUANTITY = 5              # Número de tipos de guerreros
ABILITIES = [1, 2, 3, 4, "movPos", "movNeg"]  # Acciones posibles (índices 0-3 habilidades, 4=movPos, 5=movNeg)

# ============================================================
# Juego - Recompensas
# ============================================================
REWARD_WEIGHTS = {
    "damage": 1,                  # Daño infligido (diferencia entre P1 y P2)
    "deaths": 10,                 # Muertes causadas
    "win": 1,                     # Victoria/derrota (multiplicador de WIN_REWARD)
    "blocks": 1,                  # Daño bloqueado/evadido
    "heal": 1,                    # Curación realizada
    "shaping_weight": 2,          # Peso para la diferencia de vida (shaping)
}
TURN_PENALTY = 5                  # Penalización por turno (para fomentar partidas cortas)
WIN_REWARD = 500                  # Recompensa base por ganar la partida
MAX_TURNS = 40                    # Límite de turnos por partida
MAX_DEATHS_PER_TEAM = 3           # Muertes máximas por equipo (3 = todos los guerreros)

# ============================================================
# Pool de oponentes
# ============================================================
MAX_MODELS = 50                   # Número máximo de snapshots en la pool
SAVE_MODEL_FRACTION = 0.05        # Fracción de lotes tras la cual guardar un snapshot
POOL_RANGE_FRACTION = 0.01        # Fracción de lotes tras la cual refrescar la asignación de pool
POOL_PORCENTAGE = 0.3             # Porcentaje de partidas que usan oponentes de la pool

# ============================================================
# Replay y priorización (PER)
# ============================================================
SELECTION_REPLAYS_PER_BATCH = 400  # Número de replays de selección por lote
TURN_REPLAYS_PER_BATCH = 800       # Número de replays de turno por lote

# PER (Prioritized Experience Replay)
ALPHA = 0.8                       # Factor de priorización (0=uniforme, 1=máxima prioridad)
BETA_START = 0.4                  # Factor de importancia inicial (para IS weights)
BETA_END = 1.0                    # Factor de importancia final
BETA_DECAY_RATE = 0.9999          # Decaimiento de beta por replay
PER_EPSILON = 0.1                 # Pequeño épsilon para evitar prioridades cero

# ============================================================
# N-step y arquitectura
# ============================================================
N_STEP = 1                        # Número de pasos para N-step returns (1 = estándar)
USE_DUELING_DQN = True            # Usar arquitectura Dueling en TurnNetwork
DELETE_DIRECTORIES = True         # Eliminar directorios antiguos al iniciar (para limpieza)