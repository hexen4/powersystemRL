'''
- hyperparameters
- environment
- power ratings
- cost parameters
'''

import numpy as np
from pandapower.timeseries.data_sources.frame_data import DFData
import pandas as pd


#objective func params
W1 = 0.5
W2 = 0.5

#comprehensivereplaybuffer
RHO_MIN = 10 # TODO observe and change
ETA = 0.5  
REPLAY_BUFER_SIZE = 1e6
BATCH_SIZE = 128
ALPHA = 0.6 #exponent α determines how much prioritization is used, with α = 0 corresponding to the uniform case.
BETA = 0 # importance sampling negative exponent.

#SAC only
GPU = True
DETERMINISTIC = False
device_idx = 0
DISCOUNT_FACTOR = 0.9
WEIGHT_CRITIC = 0.2
TEMP = 0.1


#ENVIRONMENT
MAX_EPISODES = 8000
HOUR_PER_TIME_STEP = 1
MAX_STEPS = 24 / HOUR_PER_TIME_STEP
WARMUP = 20 * BATCH_SIZE
UPDATE_FREQ = 5

# NN Hyperparameters
LR_ACTOR = 0.001
LR_CRITIC = 0.001
TARGET_NETWORK_UPDATE = 0.001
NN_BOUND = 1


# State
IDX_POWER_GEN = 0
IDX_SOLAR = 1
IDX_WIND = 2
IDX_CUSTOMER_PMW = np.arange(3, 8)  # Customer power: 5 indices [3, 4, 5, 6, 7]
IDX_PGRID = 8
IDX_MARKET_PRICE = 9
IDX_DISCOMFORT = np.arange(10, 15)  # Discomfort: 5 indices [10, 11, 12, 13, 14]
IDX_PREV_GENPOWER = 15
IDX_PREV_CURTAILED = np.arange(16, 21)  # Previous curtailed: 5 indices [16, 17, 18, 19, 20]
IDX_PREV_DEMAND = np.arange(21, 26)  # Previous demand: 5 indices [21, 22, 23, 24, 25]
IDX_PREV_DISCOMFORT = np.arange(26, 31)  # Previous discomfort: 5 indices [26, 27, 28, 29, 30]
IDX_LINE_LOSSES = 31    

# Action
MAX_ACTION = np.array([0.6] * 5 + [100]) #do i need to dynamically update?
MIN_ACTION = np.array([0] * 5 + [0])      # [incentive_rate_min, curtail_c1_min, ..., curtail_c5_min]
ACTION_IDX = {
    'curtail_C1': 0,               
    'curtail_C2': 1,               
    'curtail_C3': 2,               
    'curtail_C4': 3,              
    'curtail_C5': 4,               
    'incentive_rate': 5            
}

N_ACTION = len(MAX_ACTION) 
N_OBS = IDX_LINE_LOSSES + 1
ACTION_SHAPE = (N_ACTION,)
STATE_SHAPE = (N_OBS,)

# --- Cost Parameters ---
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
ids = {
'pv': 0,
'wt': 1,
'dg': 2}
for i in range(1, 33):  
    ids[f'C{i}'] = f'C{i - 1}'  # Map "C1" -> "C0", "C2" -> "C1", etc.
CONSUMER_BETA = [1,2,2,3,3]
CONSUMER_ZETA = [1,0.9,0.7,0.6,0.4]
TIMESTEPS = range(0,24)
PEAK_P_DEMAND = 3715 / 1000 #MW
PEAK_Q_DEMAND = 2300 / 1000 #MVAR
N_BUS = 33

#(bus,PL,QL)
initial_load_data = [
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

load_data = [(bus, PL / 1000, QL / 1000) for bus, PL, QL in initial_load_data]


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

pv_profile_df = pd.read_csv(filepath_results + '/pv_profile.csv')   
wt_profile_df = pd.read_csv(filepath_results + '/wt_profile.csv')
load_profile_df = pd.read_csv(filepath_results + '/load_profile.csv') / 1000 
price_profile_df = pd.read_csv(filepath_results + '/price_profile.csv')

data_source_wind = DFData(wt_profile_df)
data_source_sun = DFData(pv_profile_df)
data_source_consumers_original = DFData(load_profile_df)

model_path = r"C:\Users\jlhb83\Desktop\Python Projects\powersystemRL\model\sac_v2"



if __name__ == '__main__':
    print(f'Number of actions: {N_ACTION}')
    #print(f'Number of intermittent states: {N_INTERMITTENT_STATES}')
    #print(f'Number of controllable states: {N_CONTROLLABLE_STATES}')
    #print(f'Load max: {P_LOAD_MAX}')