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
SELECTION_REPLAY_DATA = 80_000 # Capacidad del buffer de selección Estandar 150000 parapruebas nocutrnas 80000
TURN_REPLAY_DATA = 40_000      # Capacidad del buffer de turno Estandar 150000 parapruebas nocutrnas 40000
BATCH_SIZE = 128                   # Tamaño del batch de replay
DISCOUNT_FACTOR = 0.95            # Factor de descuento (gamma)
COPY_DQN = 50                     # Frecuencia de copia a target network (en pasos de replay)

# ============================================================
# Juego - Reglas básicas
# ============================================================
WARRIOR_QUANTITY = 5              # Número de tipos de guerreros
# ============================================================
# Pool de habilidades por instancia
# ============================================================
MAX_POOL_SIZE = 6          # Tamaño de la pool de habilidades por tipo de guerrero
ABILITIES_PER_WARRIOR = 4  # Habilidades equipadas simultáneamente (no cambia)
NUM_SLOTS = 3              # Nº de slots de combate por equipo (sustituye el uso indebido de MAX_DEATHS_PER_TEAM)
# ============================================================
# Meta-juego (Castillo)
# ============================================================
USE_META_GAME = True   # Flag: True = draft desde castillo (10 slots), False = catálogo de 5 tipos

MAX_CASTLE_SIZE = 10
MAX_BATALLAS = 10
GOLD_INICIAL = 250
COST_COMPRA = 40
GOLD_POR_BATALLA = 100
MAX_ABILITY_LEVEL = 5
GOLD_NORM_REF = 500  # referencia para normalizar el oro en la observación (ajustable)

def get_selection_state_dim(use_meta: bool = None) -> int:
    """Devuelve la dimensión del estado de selección según el modo."""
    if use_meta is None:
        use_meta = USE_META_GAME
    if use_meta:
        return (
            MAX_CASTLE_SIZE * (WARRIOR_QUANTITY + ABILITIES_PER_WARRIOR * MAX_POOL_SIZE + ABILITIES_PER_WARRIOR)
            + MAX_CASTLE_SIZE   # edad por instancia
            + 1                 # oro
            + WARRIOR_QUANTITY  # one-hot del guerrero inicial del rival
            + 1                 # posición inicial del rival
        )
    else:
        return 46 + WARRIOR_QUANTITY * ABILITIES_PER_WARRIOR * MAX_POOL_SIZE

TURN_STATE_DIM = 58 + 3 * ABILITIES_PER_WARRIOR * MAX_POOL_SIZE
# ============================================================
# Juego - Recompensas
# ============================================================
REWARD_WEIGHTS = {
    "damage": 1.3,                  # Daño infligido (diferencia entre P1 y P2)
    "deaths": 50,                 # Muertes causadas
    "win": 1,                     # Victoria/derrota (multiplicador de WIN_REWARD)
    "blocks": 0.6,                  # Daño bloqueado/evadido
    "heal": 0.6,                    # Curación realizada
    "shaping_weight": 8,          # Peso para la diferencia de vida (shaping)
    "wasted_heal" : -5,
    "wasted_defense": -10,
}
# DESPUÉS
TURN_PENALTY_BASE = 2             # Penalización de turno en fase inicial (turnos <= RAMP_START)
TURN_PENALTY_RAMP_START = 4.5      # Turno a partir del cual la penalización empieza a crecer
TURN_PENALTY_RAMP_TURNS = 12      # Turnos que tarda en pasar de BASE a MAX (rampa lineal)
TURN_PENALTY_MAX = 35             # Penalización de turno una vez alcanzado el techo (cerca del límite)
WIN_REWARD = 1000               # Recompensa base por ganar la partida
# AÑADIR, junto a WIN_REWARD
DRAW_PENALTY = 200                # Penalización por resultado en empate (20% de WIN_REWARD)
MAX_TURNS = 20                    # Límite de turnos por partida
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
SELECTION_REPLAYS_PER_BATCH = 144 # Número de replays de selección por lote
TURN_REPLAYS_PER_BATCH = 480       # Número de replays de turno por lote

"""
TURN_REPLAYS_PER_BATCH_nuevo ≈ TURN_REPLAYS_PER_BATCH_actual × (avg_turns_nuevo / avg_turns_actual)
R_turn = (TURN_REPLAYS_PER_BATCH × BATCH_SIZE) / (N_BATCH × avg_turns) R estandar 3
TURN_REPLAYS_PER_BATCH = (N_BATCH × avg_turns) * R_turn / BATCH_SIZE R estandar 3
R_sel  = (SELECTION_REPLAYS_PER_BATCH × BATCH_SIZE) / (N_BATCH × 3)
SELECTION_REPLAYS_PER_BATCH = (N_BATCH × avg_turns) * R_sel / BATCH_SIZE
"""

# PER (Prioritized Experience Replay)
ALPHA = 0.8                       # Factor de priorización (0=uniforme, 1=máxima prioridad)
BETA_START = 0.4                  # Factor de importancia inicial (para IS weights)
BETA_END = 1.0                    # Factor de importancia final
BETA_DECAY_RATE = 0.9999          # Decaimiento de beta por replay
PER_EPSILON = 0.1                 # Pequeño épsilon para evitar prioridades cero

# ============================================================
# N-step y arquitectura
# ============================================================
N_STEP = 3                       # Número de pasos para N-step returns (1 = estándar) (N-STEP 5 victorias desequilibradas)
USE_DUELING_DQN = True            # Usar arquitectura Dueling en TurnNetwork
DELETE_DIRECTORIES = True         # Eliminar directorios antiguos al iniciar (para limpieza)
NOISY_SIGMA_INIT = 0.5   # Valor inicial de la desviación sigma
RESET_IN_DECISIONS = False

# ============================================================
# ELO
# ============================================================
ELO_INITIAL = 1000.0
ESTANDAR_ELO = 150
K_FACTOR_ELO = 32
ELO_TEMPERATURE = 15

# Variables que se cargarán desde YAML (con valores por defecto)
VERSION = 1
RUN_NAME_SUFFIX = ""   # se puede sobrescribir desde config.yaml
N_BATCH = 2048
TRAIN_EPISODES = 20
EVAL_EPISODES = 2
SEED = None
USE_META_GAME = True
USE_WANDB = True
RUN_COMPARISON = False
RUN_SELF_PLAY = True
RUN_EVALUATION = True
HUMAN_OPPONENT = "none"
HUMAN_EPISODES = 20
HUMAN_EPSILON = 0.1
PLAY_AGAINST_AI = False
PLAY_EPISODES = 20
PLAY_EPSILON = 0.0
