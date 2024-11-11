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

REWARD_INVALID_ACTION = -5e-3
PENALTY_FACTOR = 5
EPSILON = 0.5 #incentive per unit curtailed

WTRATED = 500 / 1000 #MW
v_opt = 12 
v_in = 3
v_coff = 25
WIND_A = WTRATED / (v_opt**3 - v_in**3)
WIND_B = v_in**3 / (v_opt**3 - v_in**3)


A1 = 0.0001 * 1000 #converted from $/kWh to $/MWh
A2 = 0.1032 * 1000 
A3 = 14.5216
PGEN_MIN = 35 / 1000 #MW
PGEN_MAX = 300 / 1000
PRAMPUP = 70 / 1000
PRAMPDOWN = 50 / 1000
MB = 0 #daily budget of MGO

NSOLAR = 4231
Ki = 0.00545
Kv = 0.1278
Tot = 45
Ta = 25
Isc = 8.95
Voc = 37.8
Impp = 8.4
Vmpp = 31
FF = (Vmpp*Impp)/(Isc*Voc)

NO_CONSUMERS = 5
CONSUMER_PARAMS = [[1,1],[2,0.9],[2,0.7],[3,0.6],[3,0.4]]
TIMESTEPS = range(0,24)
PEAK_P_DEMAND = 3715 / 1000 #MW
PEAK_Q_DEMAND = 2300 / 1000 #MVAR
N_NODE = 33
if __name__ == '__main__':
    print(f'Number of actions: {N_ACTION}')
    print(f'Number of intermittent states: {N_INTERMITTENT_STATES}')
    print(f'Number of controllable states: {N_CONTROLLABLE_STATES}')
    print(f'Load max: {P_LOAD_MAX}')