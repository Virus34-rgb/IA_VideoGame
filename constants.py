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
EPSILON_RESIDUAL = 0.01

# ============================================================
# IA - Parámetros de aprendizaje
# ============================================================
SELECTION_LEARNING_RATE = 0.0001  # Learning rate para red de selección
TURN_LEARNING_RATE = 0.0001       # Learning rate para red de turno
SELECTION_REPLAY_DATA = 1_000_000 # Capacidad del buffer de selección
TURN_REPLAY_DATA = 1_000_000      # Capacidad del buffer de turno
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
    "damage": 0,                  # Daño infligido (diferencia entre P1 y P2)
    "deaths": 50,                 # Muertes causadas
    "win": 1,                     # Victoria/derrota (multiplicador de WIN_REWARD)
    "blocks": 0.01,                  # Daño bloqueado/evadido
    "heal": 0.01,                    # Curación realizada
    "shaping_weight": 2,          # Peso para la diferencia de vida (shaping)
}
TURN_PENALTY = 50                # Penalización por turno (para fomentar partidas cortas)
WIN_REWARD = 1000                  # Recompensa base por ganar la partida
MAX_TURNS = 25                   # Límite de turnos por partida
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
SELECTION_REPLAYS_PER_BATCH = 1024  # Número de replays de selección por lote
TURN_REPLAYS_PER_BATCH = 2048       # Número de replays de turno por lote

# PER (Prioritized Experience Replay)
ALPHA = 0.8                       # Factor de priorización (0=uniforme, 1=máxima prioridad)
BETA_START = 0.4                  # Factor de importancia inicial (para IS weights)
BETA_END = 1.0                    # Factor de importancia final
BETA_DECAY_RATE = 0.9999          # Decaimiento de beta por replay
PER_EPSILON = 0.1                 # Pequeño épsilon para evitar prioridades cero

# ============================================================
# N-step y arquitectura
# ============================================================
N_STEP = 3                        # Número de pasos para N-step returns (1 = estándar)
USE_DUELING_DQN = True            # Usar arquitectura Dueling en TurnNetwork
DELETE_DIRECTORIES = True         # Eliminar directorios antiguos al iniciar (para limpieza)
NOISY_SIGMA_INIT = 0.2   # Valor inicial de la desviación sigma
RESET_IN_DECISIONS = True

# ============================================================
# ELO
# ============================================================
ELO_INITIAL = 1000.0
ESTANDAR_ELO = 400
K_FACTOR_ELO = 32
ELO_TEMPERATURE = 150
