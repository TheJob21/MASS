import torch

# GENERAL EXECUTION SETTINGS
SIM_MODE = True          # True = synthetic, False = live spectrum
MULTI_AGENT = True
EVAL_MODE = False
RANDOM_START_INDICES = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42069

CHECKPOINT_DIR = "Agents/Checkpoints"
LOAD_CHECKPOINTS = False

AUTO_SAVE_LATEST = False


OUTPUT_FILE = "./Output/agent_eval_summary_multiUnion245.xlsx"

# DATA_CHOICE = "245"
DATA_CHOICE = "u245"
# DATA_CHOICE = "264"
# DATA_CHOICE = "u264"

SPECTRUM_FILES = {
    "245": "./Data/spectrum_245ghz.dat",
    "u245": "./Data/union_spectrum_245ghz.dat",
    "264": "./Data/spectrum_264ghz.dat",
    "u264": "./Data/union_spectrum_264ghz.dat"
}

STORED_STATE_MAP = {
    "./Data/spectrum_245ghz.dat": "./Data/spectrum_245ghz.npz",
    "./Data/union_spectrum_245ghz.dat": "./Data/union_spectrum_245ghz.npz",
    "./Data/spectrum_264ghz.dat": "./Data/spectrum_264ghz.npz",
    "./Data/union_spectrum_264ghz.dat": "./Data/union_spectrum_264ghz.npz"
}

# SPECTRUM / SIGNAL PROCESSING
FFT_SIZE = 1024
CHANNEL_BANDWIDTH = 100  # MHz
BIN_SIZE = CHANNEL_BANDWIDTH / FFT_SIZE

PRI = 204.8
CPI_LEN = 256

STARTING_FREQUENCY_MAP = {
    "./Data/spectrum_245ghz.npz": 2400,
    "./Data/union_spectrum_245ghz.npz": 2400,
    "./Data/spectrum_264ghz.npz": 2590,
    "./Data/union_spectrum_264ghz.npz": 2590
}

HOCAE_WINDOW_SIZE = 32
HOCAE_ORDER_SELECTION = 5
HOCAE_PFA = 1e-2

# AGENT COUNTS
AGENTS = {
    "static": {
        "fat": 2,
        "skinny": 3,
        "pulsed": 5,
        "rectangular": 0
    },
    "random_start": 0,
    "saa": 0,
    "ppo": 0,
    "dqn": 0,
    "mfos": 1,
    "dpg": 0,
    "ablated_mfos": 1
}

# PPO
PPO = {
    "time_horizon": 1024,
    "num_actors": 16,
    "gamma": 0.8,
    "gae_lambda": 0.95,
    "clip": 0.2,
    "epochs": 10,
    "lr": 2.5e-4
}

# DQN
DQN = {
    "memory_size": 2000,
    "batch_size": 32,
    "gamma": 0.9,
    "lr": 1e-3,
    "target_update": 250,
    "hidden_layers": [256, 128, 84]
}

# MFOS
MFOS = {
    "population_size": 5,
    "mutation_scale": 0.05,
    "elite_fraction": 0.4,
    "fresh_fraction": 0.2
}

# REWARD FUNCTION PARAMETERS

# collisionTransmissionTolRatio = 0.0125 # for pulsed aversions
# collisionTransmissionTolRatio = 0.33 # for constant aversions Use worst reward for pulses, not effective in 2.4-2.5GHz live data
# collisionTransmissionTolRatio = 0.08 # effective in 2.4-2.5GHz live data,  Use worst reward for pulses
# collisionTransmissionTolRatio = 0.04 # effective in 2.59-2.69GHz live data,  Use worst reward for pulses
# collisionTransmissionTolRatio = 0.033 # Shane's recommendation 30 * collision
# collisionTransmissionTolRatio = 0.0355
# collisionWeight = 29
# collisionTransmissionTolRatio = .0275
# collisionTransmissionTolRatio = 0.033

REWARD = {
    "collision_ratio": 0.02,#0.033,
    "beta": 0.75,
    "transmission_weight": 1.0,
    "collision_weight": None,  # computed dynamically if needed
    "bandwidth_distortion": 0.3,
    "center_distortion": 0.3,
    "deadspace_penalty_scale": 1.0
}

REWARD["collision_weight"] = (
    REWARD["transmission_weight"] / REWARD["collision_ratio"]
)

LIMIT_OBSERVATION = True
OBSERVATION_BIN_SIZE = 300
if not LIMIT_OBSERVATION:
    OBSERVATION_BIN_SIZE = FFT_SIZE

OBSERVATION_CENTER_COUNT = 3

# RUNTIME CONTROL
ITERATIONS = 5_000_000
SPECTRUM_SAMPLE_SIZE = 15_000

EVAL_SPLIT = 0.8
PRINT_INTERVAL = 100_000

TIMESTEP_US = 10.24