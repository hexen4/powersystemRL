'''
- hyperparameters
- environment
- power ratings
- cost parameters
'''

import numpy as np
from pandapower.timeseries.data_sources.frame_data import DFData
import pandas as pd
# --- Hyperparameters ---
BATCH_SIZE = 128
GAMMA = 0.99
LR_ACTOR = 0.001
LR_CRITIC = 0.001
NN_BOUND = 1
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

# P_PV3_MAX = 0.3
# P_PV4_MAX = 0.3
# P_PV5_MAX = 0.4
# P_PV6_MAX = 0.4
# P_PV8_MAX = 0.4
# P_PV9_MAX = 0.5
# P_PV10_MAX = 0.5
# P_PV11_MAX = 0.3
# P_PV_MAX_LIST = [P_PV3_MAX, P_PV4_MAX, P_PV5_MAX, P_PV6_MAX, P_PV8_MAX, P_PV9_MAX, P_PV10_MAX, P_PV11_MAX]

# # WT
# P_WT7_MAX = 2.5
# P_WT_MAX_LIST = [P_WT7_MAX]

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
# E_B5_MAX = 3.
# P_B5_MAX = 0.6
# P_B5_MIN = -0.6

# E_B10_MAX = 1.
# P_B10_MAX = 0.2
# P_B10_MIN = -0.2

# SOC_MAX = 0.9
# SOC_MIN = 0.1
# SOC_TOLERANCE = 0.01

# # Load
# P_LOADR1_MAX = 0.85
# P_LOADR3_MAX = 0.285
# P_LOADR4_MAX = 0.245
# P_LOADR5_MAX = 0.65
# P_LOADR6_MAX = 0.565
# P_LOADR8_MAX = 0.605
# P_LOADR10_MAX = 0.49
# P_LOADR11_MAX = 0.34
# P_LOAD_MAX_LIST = [P_LOADR1_MAX, P_LOADR3_MAX, P_LOADR4_MAX, P_LOADR5_MAX, P_LOADR6_MAX, P_LOADR8_MAX, P_LOADR10_MAX, P_LOADR11_MAX]
# P_LOAD_MAX = P_LOADR1_MAX + P_LOADR3_MAX + P_LOADR4_MAX + P_LOADR5_MAX + P_LOADR6_MAX + P_LOADR8_MAX + P_LOADR10_MAX + P_LOADR11_MAX

# # PCC
# P_EXCESS_MAX = sum([*P_PV_MAX_LIST, *P_WT_MAX_LIST])

# State
IDX_POWER_GEN = 0
IDX_SOLAR = 1
IDX_WIND = 2
IDX_CUSTOMER_PMW = 3
IDX_PGRID = 4
IDX_MARKET_PRICE = 5
IDX_DISCOMFORT = 6
IDX_PREV_POWER = 7
IDX_PREV_CURTAILED = 8
IDX_PREV_DEMAND = 9
IDX_PREV_DISCOMFORT = 10
# N_INTERMITTENT_STATES = len([P_EXCESS_MAX,'price'])
#N_INTERMITTENT_STATES = len([*P_PV_MAX_LIST, *P_WT_MAX_LIST, *P_LOAD_MAX_LIST,'price'])
# N_INTERMITTENT_STATES = len([*P_PV_MAX_LIST, *P_WT_MAX_LIST, *P_LOAD_MAX_LIST, P_EXCESS_MAX,'price'])
#N_CONTROLLABLE_STATES = len([P_B5_MAX, P_B10_MAX]) # TODO change the states here
# TODO add uncertainty power output of wind + PV
#STATE_SEQ_SHAPE = (SEQ_LENGTH, N_INTERMITTENT_STATES)
#STATE_FNN_SHAPE = (N_CONTROLLABLE_STATES,)

# Action
MAX_ACTION = np.array([100] + [600] * 5)  # [incentive_rate_max, curtail_c1_max, ..., curtail_c5_max]
MIN_ACTION = np.array([0] + [0] * 5)      # [incentive_rate_min, curtail_c1_min, ..., curtail_c5_min]
ACTION_IDX = {
    'incentive_rate': 0,          # Incentive rate action at index 0
    'curtail_C1': 1,               # Curtailment for consumer 1 at index 1
    'curtail_C2': 2,               # Curtailment for consumer 2 at index 2
    'curtail_C3': 3,               # Curtailment for consumer 3 at index 3
    'curtail_C4': 4,               # Curtailment for consumer 4 at index 4
    'curtail_C5': 5                # Curtailment for consumer 5 at index 5
}
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
CONSUMER_BETA = [1,2,2,3,3]
CONSUMER_ZETA = [1,0.9,0.7,0.6,0.4]
TIMESTEPS = range(0,24)
PEAK_P_DEMAND = 3715 / 1000 #MW
PEAK_Q_DEMAND = 2300 / 1000 #MVAR
N_NODE = 33
#(bus,PL,QL)
load_data = [
(1, 100, 60),
(2, 90, 40),
(3, 120, 80),
(4, 60, 30),
(5, 60, 20),
(6, 200, 100),
(7, 200, 100),
(8, 60, 20),
(9, 60, 20),
(10, 45, 30),
(11, 60, 35),
(12, 60, 35),
(13, 120, 80),
(14, 60, 10),
(15, 60, 20),
(16, 60, 20),
(17, 90, 40),
(18, 90, 40),
(19, 90, 40),
(20, 90, 40),
(21, 90, 40),
(22, 90, 50),
(23, 420, 200),
(24, 420, 200),
(25, 60, 25),
(26, 60, 25),
(27, 60, 20),
(28, 120, 70),
(29, 200, 600),
(30, 150, 70),
(31, 210, 100),
(32, 60, 40)]

#(to_bus, from_bus, r_ohm(total), x_ohm)
line_data = [
(0, 1, 0.0922, 0.0470), ##SS
(1, 2, 0.4930, 0.2511),
(2, 3, 0.3660, 0.1864),
(3, 4, 0.3811, 0.1941),
(4, 5, 0.8190, 0.7070),
(5, 6, 0.1872, 0.6188),
(6, 7, 1.7114, 0.2351),
(7, 8, 1.0300, 0.7400),
(8, 9, 1.0400, 0.7400),
(9, 10, 0.1966, 0.0650),
(10, 11, 0.3744, 0.1238),
(11, 12, 1.4680, 0.1550),
(12, 13, 0.5416, 0.7129),
(13, 14, 0.5910, 0.5260),
(14, 15, 0.7463, 0.5450),
(15, 16, 1.2890, 1.7210),
(16, 17, 0.7320, 0.5740),
(18, 1, 0.1640, 1.1565),
(18, 19, 1.5402, 1.3554),
(19, 20, 0.4095, 0.4784),
(20, 21, 0.7089, 0.9373),
(2, 22, 0.4512, 0.3083),
(22, 23, 0.8980, 0.7091),
(23, 24, 0.8960, 0.7011),
(5, 25, 0.2030, 0.1034),
(25, 26, 0.2842, 0.1447),
(26, 27, 1.0590, 0.9337),
(27, 28, 0.8042, 0.7006),
(28, 29, 0.5075, 0.2585),
(29, 30, 0.9744, 0.9630),
(30, 31, 0.3105, 0.3619),
(31, 32, 0.3410, 0.5302)]

filepath_results = r"C:\Users\jlhb83\Desktop\Python Projects\powersystemRL\data\derived"
power_data_path_wind = r"C:\Users\jlhb83\Desktop\Python Projects\powersystemRL\data\derived\wt_profile.csv"
power_data_path_sun = r"C:\Users\jlhb83\Desktop\Python Projects\powersystemRL\data\derived\pv_profile.csv"
power_data_consumers =r"C:\Users\jlhb83\Desktop\Python Projects\powersystemRL\data\derived\load_profile.csv"

datasource_wind = pd.read_csv(power_data_path_wind)
datasource_sun = pd.read_csv(power_data_path_sun)
datasource_consumers = pd.read_csv(power_data_consumers) * PEAK_P_DEMAND / 100 #individual consumer profiles in percentage

data_source_wind = DFData(datasource_wind)
data_source_sun = DFData(datasource_sun)
data_source_consumers = DFData(datasource_consumers)

pv_profile_df = pd.read_csv(filepath_results + '/pv_profile.csv') # TODO is this correct at this place
wt_profile_df = pd.read_csv(filepath_results + '/wt_profile.csv')
load_profile_df = pd.read_csv(filepath_results + '/load_profile.csv')
price_profile_df = pd.read_csv(filepath_results + '/price_profile.csv')

if __name__ == '__main__':
    print(f'Number of actions: {N_ACTION}')
    #print(f'Number of intermittent states: {N_INTERMITTENT_STATES}')
    #print(f'Number of controllable states: {N_CONTROLLABLE_STATES}')
    #print(f'Load max: {P_LOAD_MAX}')