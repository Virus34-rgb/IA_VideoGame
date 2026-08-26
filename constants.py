# CastleGame/constants.py

#IA
EPSILON_SELECTION = 0.5
EPSILON_TURN = 0.5
EPSILON_SEL_MIN = 0.05
EPSILON_TURN_MIN = 0.05
EPSILON_SEL_DECAY = 0.99998
EPSILON_TURN_DECAY = 0.99998
SELECTION_LEARNING_RATE = 0.0001
TURN_LEARNING_RATE = 0.0001
SELECTION_REPLAY_DATA = 100_000
TURN_REPLAY_DATA = 100_000
BATCH_SIZE = 64
DISCOUNT_FACTOR = 0.95  #Base 0.95
COPY_DQN = 50

#BASE DEL JUEGO
WARRIOR_QUANTITY = 5
ABILITIES = [1, 2, 3, 4, "movPos", "movNeg"]

# --- Pesos de la función de recompensa ---
REWARD_WEIGHTS = {
    "damage": 1,
    "deaths": 10,
    "win": 1,       # WIN_REWARD ya se aplica dentro del bonus, este peso multiplica ese bonus
    "blocks": 1,
    "heal":1,
    "shaping_weight":10,
}
TURN_PENALTY = 2
WIN_REWARD = 100
MAX_TURNS = 80
MAX_DEATHS_PER_TEAM = 3

#CONSTANTES PARA LA POOL DE ENEMIGOS
MAX_MODELS = 50
SAVE_MODEL_FRACTION = 0.05   # ~5% de los lotes del run
POOL_RANGE_FRACTION = 0.01   # ~1% de los lotes del run
POOL_PORCENTAGE = 0.3
SELECTION_REPLAYS_PER_BATCH = 20   # antes eran ~1536, ajusta empezando bajo
TURN_REPLAYS_PER_BATCH = 40  

#CONSTANTES PER
ALPHA = 0.8 #A mayor nummero mas prioridad se le da a a la prioridad 1.0 MAS AGRESIVO 0.6 MAS CONSERVADOR
BETA_START = 0.4
BETA_END = 1.0
BETA_DECAY_RATE = 0.9999
PER_EPSILON = 0.1

#CONSTANTES N-STEPS
N_STEP = 3