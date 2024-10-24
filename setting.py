'''
- hyperparameters
- environment
- power ratings
- cost parameters
'''

import numpy as np

# --- Hyperparameters ---
BATCH_SIZE = 128
GAMMA = 0.99
LR_ACTOR = 0.001
LR_CRITIC = 0.001
NN_BOUND = 1.
SEQ_LENGTH= 1

# TD3 only
ACTION_NOISE_SCALE = 0.3
BUFFER_SIZE = 500000
NOISE_TYPE = 'param' # ['action', 'param']
PARAM_NOISE_ADAPT_RATE = 1.01
PARAM_NOISE_BOUND = 0.1
PARAM_NOISE_SCALE = 0.1
UPDATE_FREQ = 50
UPDATE_TIMES = 4
WARMUP = 1000

# PPO only
POLICY_CLIP = 0.2
TARGET_KL = 0.01
PPO_BATCH_SIZE = 60
PPO_TRAIN_FREQ = 720
PPO_TRAIN_ITERS = 80

# others
PREDICT_LENGTH = 24
DENSE_DIM_A = 16
DENSE_DIM_FNN = 16
DENSE_DIM_SEQ = 32

# Environment
HOUR_PER_TIME_STEP = 1

# --- Power Ratings ---
# PV
# TODO replace this with uncertain values?
P_PV3_MAX = 0.3
P_PV4_MAX = 0.3
P_PV5_MAX = 0.4
P_PV6_MAX = 0.4
P_PV8_MAX = 0.4
P_PV9_MAX = 0.5
P_PV10_MAX = 0.5
P_PV11_MAX = 0.3
P_PV_MAX_LIST = [P_PV3_MAX, P_PV4_MAX, P_PV5_MAX, P_PV6_MAX, P_PV8_MAX, P_PV9_MAX, P_PV10_MAX, P_PV11_MAX]

# WT
P_WT7_MAX = 2.5
P_WT_MAX_LIST = [P_WT7_MAX]

# MGT
# P_MGT5_MAX = 0.033
# P_MGT9_MAX = 0.212
# P_MGT10_MAX = 0.033
# P_MGT5_MIN = 0.
# P_MGT9_MIN = 0.
# P_MGT10_MIN = 0.
# P_MGT_MAX_LIST = [P_MGT5_MAX, P_MGT9_MAX, P_MGT10_MAX]
 
# Battery
# TODO ask Pratik about the power ratings
E_B5_MAX = 3.
P_B5_MAX = 0.6
P_B5_MIN = -0.6

E_B10_MAX = 1.
P_B10_MAX = 0.2
P_B10_MIN = -0.2

SOC_MAX = 0.9
SOC_MIN = 0.1
SOC_TOLERANCE = 0.01

# Load
P_LOADR1_MAX = 0.85
P_LOADR3_MAX = 0.285
P_LOADR4_MAX = 0.245
P_LOADR5_MAX = 0.65
P_LOADR6_MAX = 0.565
P_LOADR8_MAX = 0.605
P_LOADR10_MAX = 0.49
P_LOADR11_MAX = 0.34
P_LOAD_MAX_LIST = [P_LOADR1_MAX, P_LOADR3_MAX, P_LOADR4_MAX, P_LOADR5_MAX, P_LOADR6_MAX, P_LOADR8_MAX, P_LOADR10_MAX, P_LOADR11_MAX]
P_LOAD_MAX = P_LOADR1_MAX + P_LOADR3_MAX + P_LOADR4_MAX + P_LOADR5_MAX + P_LOADR6_MAX + P_LOADR8_MAX + P_LOADR10_MAX + P_LOADR11_MAX

# PCC
P_EXCESS_MAX = sum([*P_PV_MAX_LIST, *P_WT_MAX_LIST])

# State
# N_INTERMITTENT_STATES = len([P_EXCESS_MAX,'price'])
N_INTERMITTENT_STATES = len([*P_PV_MAX_LIST, *P_WT_MAX_LIST, *P_LOAD_MAX_LIST,'price'])
# N_INTERMITTENT_STATES = len([*P_PV_MAX_LIST, *P_WT_MAX_LIST, *P_LOAD_MAX_LIST, P_EXCESS_MAX,'price'])
N_CONTROLLABLE_STATES = len([P_B5_MAX, P_B10_MAX]) # TODO change the states here
# TODO add uncertainty power output of wind + PV
STATE_SEQ_SHAPE = (SEQ_LENGTH, N_INTERMITTENT_STATES)
STATE_FNN_SHAPE = (N_CONTROLLABLE_STATES,)

# Action
# TODO actions only defined for batteries?
ACTION_IDX = {'p_b5': 0, 'p_b10': 1}
MAX_ACTION = np.array([P_B5_MAX, P_B10_MAX])
MIN_ACTION = np.array([P_B5_MIN, P_B10_MIN])
N_ACTION = len(MAX_ACTION)

# --- Cost Parameters ---
# TODO change cost parameters
C_PRICE_MAX = 3.
# C_MGT5 = [100, 1.5]
# C_MGT9 = [15.8, 2.]
# C_MGT10 = [100, 1.5]
C_BAT5_DoD = 0.43
C_BAT10_DoD = 0.16
C_SOC_LIMIT = 100
MAX_COST = C_PRICE_MAX * (P_B5_MAX + P_B10_MAX + P_LOAD_MAX) + \
        (C_BAT5_DoD + C_BAT10_DoD) * pow(SOC_MAX-SOC_MIN, 2) + \
        C_SOC_LIMIT

REWARD_INVALID_ACTION = -5e-3

if __name__ == '__main__':
    print(f'Number of actions: {N_ACTION}')
    print(f'Number of intermittent states: {N_INTERMITTENT_STATES}')
    print(f'Number of controllable states: {N_CONTROLLABLE_STATES}')
    print(f'Load max: {P_LOAD_MAX}')