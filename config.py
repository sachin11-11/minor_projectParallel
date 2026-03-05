# Configuration for DQN Traffic Light Control

# Training parameters
EPISODES = 100
MAX_STEPS_PER_EPISODE = 3600  # 1 hour of simulation flow data

# Checkpoint and evaluation intervals
CHECKPOINT_INTERVAL = 2   # Save checkpoint every 2 episodes
TEST_INTERVAL = 50         # Test agent every 50 episodes

# DQN Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
GAMMA = 0.95  # Discount factor
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99
REPLAY_BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 10  # Update target network every N episodes

# Parallel training parameters
NUM_WORKERS = 4           # Number of parallel SUMO worker processes
WEIGHT_SYNC_INTERVAL = 50 # Broadcast weights to workers every N training steps

# Traffic light control parameters
YELLOW_PHASE_DURATION = 3  # seconds
GREEN_PHASE_DURATION = 15  # seconds
ACTION_DURATION = GREEN_PHASE_DURATION  # Agent makes decisions every 15 seconds

# SUMO configuration
SUMO_CONFIG_PATH = "Enviroment/simulation.sumocfg"
SUMO_GUI = False  # Set to True to visualize during training

# Flow files for Day 1 only
FLOW_FILES = [
    "Enviroment/flows_day_1.xml",
]

# Traffic light IDs
TLS_IDS = {
    "maitighar_main": "clusterJ41_J42_J43",
    "maitighar_right": "clusterJ39_J40",
    "kupondole": "clusterJ26_J58",
    "tripureshwor": "clusterJ55_clusterJ51_J53",
    "maternity": "clusterJ44_J47"
}

# Phase definitions
PHASES = {
    0: "PHASE_1_STRAIGHTS",      # Maitighar <-> Kupondole
    1: "PHASE_2_MAITIGHAR_RIGHT", # Maitighar -> Tripureshwor + Kupondole
    2: "PHASE_3_TRIPURESHWOR_STRAIGHT", # Tripureshwor -> Maternity + Left
    3: "PHASE_4_KUPONDOLE_RIGHT"  # Kupondole -> Maternity + Maitighar
}

NUM_ACTIONS = len(PHASES)

# Model save paths
MODEL_SAVE_PATH = "trained_models/dqn_traffic_light.pth"
CHECKPOINT_DIR = "trained_models/checkpoints"
PLOT_SAVE_PATH = "results/training_rewards.png"

# Graph output directories
GRAPH_DIR = "GRAPH"
REWARD_GRAPH_DIR = "GRAPH/REWARD_GRAPH"
