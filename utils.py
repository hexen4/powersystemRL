"""
Function List:
- normalize_df_column
- scale_action
- update_mean_var_count_from_moments
- NormalizeAction
    - normalize
    - tf_normalize
- NormalizeObservation
    - normalize
    - save
    - load
- NormalizeReward
    - normalize
- RunningMeanStd
    - update
    - update_from_moments
- normalize_state
- extra_reward
- plot_ep_values
- plot_pf_results
- plot_results
- log_actor_critic_info
- log_calc_rewards
- log_trans_info
- get_excess
- policy_simple
- calculate_shape_parameters
- calculate_f_wind
- calculate_chs
- calculate_khs
- calculate_f_solar
- calculate_wind_power
- calculate_solar_power
- create_output_writer
- calculate_discomfort
"""


import os
from pandapower.timeseries import OutputWriter
import logging
import pickle
from pathlib import Path
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd
#import tensorflow as tf
#from scipy.special import gamma
from setting import *
# Configure logging to write to a file
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Set the format of the log messages
    handlers=[
        logging.FileHandler("rewards_log.txt"),  # Write logs to a file
        logging.StreamHandler()  # Optionally, also print logs to the console
    ]
)

def normalize_df_column(df, column_name):
    col_min = df[column_name].min()
    col_max = df[column_name].max()
    return (df[column_name] - col_min) / (col_max - col_min)


# --- Action Scaling --- [0,1] -> [min, max]
def scale_action(nn_action, max_action, min_action=MIN_ACTION):
    action = min_action + (nn_action + 1.0) * 0.5 * (max_action - min_action)
    action = np.clip(action, min_action, max_action)
    return action
# --- Normalization ---
def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class NormalizeObservation:
    def __init__(self, shape, epsilon=1e-8):
        self.obs_rms = RunningMeanStd(shape=shape)  
        self.epsilon = epsilon

    def normalize(self, state, update=True):
        if update:
            self.obs_rms.update(state)
        normalized_state = (state - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        normalized_state = np.clip(normalized_state, -5, 5)
        return normalized_state


    def save(self, dir):
        fpath = Path(os.path.join(dir, 'obs.pkl'))
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(fpath, 'wb') as f:
            pickle.dump({
                'obs_mean': self.obs_rms.mean,
                'obs_var': self.obs_rms.var,
            }, f)

    def load(self, dir):
        with open(os.path.join(dir, 'obs.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.obs_rms.mean = data['obs_mean']
            self.obs_rms.var = data['obs_var']


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)




# --- Plot ---
def plot_ep_values(ep_values, train_length, epochs, ylabel):
    runs = ep_values.shape[0]
    fig_path = os.path.join('plot', f'{int(train_length/24)}days_{runs}runs_{epochs}eps_{str.lower(ylabel)}.png')
    arr_path = os.path.join('plot', f'{int(train_length/24)}days_{runs}runs_{epochs}eps_{str.lower(ylabel)}.npy')
    np.save(arr_path, ep_values)

    ep_return = np.median(ep_values, axis=0)
    epochs = range(1, len(ep_return) + 1)
    plt.plot(epochs, ep_return)
    plt.title(f'Training')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.savefig(fig_path)
    plt.show()

def plot_pf_results(dir, start, length):
    # pv, wt, mgt, load, bat, util, excess
    res_sgen_file = os.path.join(dir, 'res_sgen', 'p_mw.csv')
    res_load_file = os.path.join(dir, 'res_load', 'p_mw.csv')
    res_storage_file = os.path.join(dir, 'res_storage', 'p_mw.csv')
    res_trafo_file = os.path.join(dir, 'res_trafo', 'p_lv_mw.csv')

    # pv, wt, mgt
    sgen_p_mw = pd.read_csv(res_sgen_file)
    pv_p_mw = sgen_p_mw.iloc[:, 1:9]
    pv_p_mw.columns = ['pv3', 'pv4', 'pv5', 'pv6', 'pv8', 'pv9', 'pv10', 'pv11']
    wt_p_mw = sgen_p_mw.iloc[:, [9]]
    wt_p_mw.columns = ['wt7']
    # mgt_p_mw = sgen_p_mw.iloc[:, 10:]
    # mgt_p_mw.columns = ['mgt5', 'mgt9', 'mgt10']

    # load
    load_p_mw = pd.read_csv(res_load_file)
    load_p_mw = load_p_mw.iloc[:, 1:]
    load_p_mw.columns = ['load_r1', 'load_r3', 'load_r4', 'load_r5', 'load_r6', 'load_r8', 'load_r10', 'load_r11']

    # bat
    bat_p_mw = pd.read_csv(res_storage_file)
    bat_p_mw = bat_p_mw.iloc[:, 1:]
    bat_p_mw.columns = ['bat5', 'bat10']

    # utility
    trafo_p_mw = pd.read_csv(res_trafo_file)
    util_p_mw = -trafo_p_mw.iloc[:, [1]]
    util_p_mw.columns = ['utility']

    # price
    price = pd.read_csv(os.path.join('.', 'data', 'profile', 'price_profile.csv'))

    excess_p_mw = pv_p_mw.sum(axis=1) + wt_p_mw.sum(axis=1) - load_p_mw.sum(axis=1)
    excess_p_mw = pd.DataFrame({'excess': excess_p_mw})

    ax = excess_p_mw.iloc[start: start+length].plot(drawstyle='steps-post')
    bat_p_mw.iloc[start: start+length].plot(ax=ax, drawstyle='steps-post')
    price.iloc[start: start+length].plot(ax=ax, drawstyle='steps-post')
    plt.title('Power Flow')
    plt.xlabel('hour')
    plt.ylabel('MW')
    plt.show()
def plot_results(output_dir):
    # Plot load power results
    res_load_file = os.path.join(output_dir, "res_load", "p_mw.xlsx")
    res_load = pd.read_excel(res_load_file, index_col=0)
    res_load.plot()
    plt.xlabel("Time step")
    plt.ylabel("Load power [MW]")
    plt.title("Load Power Over Time")
    plt.grid()
    plt.show()

    # Plot voltage magnitude results
    res_bus_vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.xlsx")
    res_bus_vm_pu = pd.read_excel(res_bus_vm_pu_file, index_col=0)
    res_bus_vm_pu.plot()
    plt.xlabel("Time step")
    plt.ylabel("Voltage magnitude [p.u.]")
    plt.title("Voltage Magnitude Over Time")
    plt.grid()
    plt.show()

    # Plot generator power results
    res_gen_file = os.path.join(output_dir, "res_gen", "p_mw.xlsx")
    res_gen = pd.read_excel(res_gen_file, index_col=0)
    res_gen.plot()
    plt.xlabel("Time step")
    plt.ylabel("Generator Power [MW]")
    plt.title("Generator Power Over Time")
    plt.legend(['WT1','CDG1'],loc='upper right')
    plt.grid()
    plt.show()

    # Plot static generator (sgen) power results
    res_sgen_file = os.path.join(output_dir, "res_sgen", "p_mw.xlsx")
    res_sgen = pd.read_excel(res_sgen_file, index_col=0)
    res_sgen.plot()
    plt.xlabel("Time step")
    plt.ylabel("Solar sGen Power [MW]")
    plt.title("Solar Gen Power Over Time")
    plt.grid()
    plt.show()

    # plt.bar(TIMESTEPS, load_profile_df['C1'], label='C1', color='blue', bottom=None)
    # plt.bar(TIMESTEPS, load_profile_df['C2'], label='C2', color='orange', bottom=load_profile_df['C1'])
    # plt.bar(TIMESTEPS, load_profile_df['C3'], label='C3', color='purple', bottom=load_profile_df['C1'] + load_profile_df['C2'])
    # plt.bar(TIMESTEPS, load_profile_df['C4'], label='C4', color='green', bottom=load_profile_df['C1'] + load_profile_df['C2'] + load_profile_df['C3'])
    # plt.bar(TIMESTEPS, load_profile_df['C5'], label='C5', color='red', bottom=load_profile_df['C1'] + load_profile_df['C2'] + load_profile_df['C3'] + load_profile_df['C4'])
    # plt.legend(title="Consumers", loc="upper left")
    # plt.xlabel("Time step")
    # plt.ylabel("Load Power [MW]")
    # plt.title("Load Power Over Time")
    # plt.show()
    price_profile_df.plot(xlabel='hour', ylabel='price', title='Price')
    plt.show()


# --- Logging ---
def log_actor_critic_info(actor_loss, critic_loss, t=None, freq=20, **kwargs):
    if t is None:
        logging.info('--- Learn ---')
        logging.info(f'actor loss = {actor_loss}')
        logging.info(f'critic loss = {critic_loss}')
        return

    if t % freq == 0:
        logging.info('--- Learn ---')
        logging.info(f'actor loss = {actor_loss}')
        logging.info(f'critic loss = {critic_loss}')

def log_calc_rewards(t, source='', freq=5, penalties=None, reward=None, scaled_action=None, state=None, **kwargs):
    """
    Logs penalties, profit, and reward information at specified time intervals.
    """
    # Define the mapping for state indices
    state_names = {
        0: "POWER_GEN",
        1: "PREV_GEN_COST",
        2: "MARKET_PRICE",
        3: "PGRID",
        4: "PREV_POWER_TRANSFER_COST",
        5: "PREV_MGO_PROFIT",
        6: "SOLAR",
        7: "WIND",
        8: "TOTAL_LOAD",
        9: "LINE_LOSSES",
        10: "PREV_GENPOWER",
    }

    # Add ranges for indexed consumers (5 consumers in this case)
    state_names.update({i: f"ACTIVE_PMW_CONSUMER_{i-11}" for i in range(11, 16)})
    state_names.update({i: f"PREV_CURTAILED_CONSUMER_{i-16}" for i in range(16, 21)})
    state_names.update({i: f"PREV_ACTIVE_PMW_CONSUMER_{i-21}" for i in range(21, 26)})
    state_names.update({i: f"DISCOMFORT_CONSUMER_{i-26}" for i in range(26, 31)})
    state_names.update({i: f"PREV_ACTIVE_BENEFIT_CONSUMER_{i-31}" for i in range(31, 36)})

    # Add remaining state indices
    state_names[36] = "MINMARKET_PRICE"
    state_names[37] = "PREV_BUDGET"
    if t % freq == 0:


        # Log reward, profit, and penalties
        logging.info(f'--- {source} ---')
        logging.info(f'Reward: {reward:.3f}')
        logging.info(f'--- Penalties and Profit ---')
        for penalty_name, penalty_value in penalties.items():
            logging.info(f'{penalty_name}: {penalty_value:.3f}')
        if scaled_action is not None:
            logging.info(f"Scaled Action: {scaled_action}")

        logging.info(f"time step: {t}")

        # Log state
        if state is not None:
            logging.info("State:")
            for idx, value in enumerate(state):
                state_name = state_names.get(idx, f"IDX_{idx}") 
                logging.info(f"  {state_name}: {value}")
def log_trans_info(s, a, t, freq=100, **kwargs):
    if t % freq == 0:
        s_seq = s[0]    
        s_fnn = s[1]

        logging.info('--- State ---')
        logging.info(f'shape: ({s_seq.shape}, {s_fnn.shape})')
        logging.info(f'content: {s_seq[0]}, {s_fnn}')
        logging.info('--- Action ---')
        logging.info(f'shape: {a.shape}')
        logging.info(f'content: {a}')

# --- Others ---
def get_excess(pv_profile, wt_profile, load_profile, t):
    excess = pv_profile['pv3'][t] +\
        pv_profile['pv4'][t] +\
        pv_profile['pv5'][t] +\
        pv_profile['pv6'][t] +\
        pv_profile['pv8'][t] +\
        pv_profile['pv9'][t] +\
        pv_profile['pv10'][t] +\
        pv_profile['pv11'][t] +\
        wt_profile['wt7'][t] -\
        load_profile['load_r1'][t] -\
        load_profile['load_r3'][t] -\
        load_profile['load_r4'][t] -\
        load_profile['load_r5'][t] -\
        load_profile['load_r6'][t] -\
        load_profile['load_r8'][t] -\
        load_profile['load_r10'][t] -\
        load_profile['load_r11'][t]

    return excess


def calculate_shape_parameters(mu_h_wind, sigma_h_wind):
    """
    Calculate the shape parameters k_w^h and c_w^h for the given mean (mu_h_wind)
    and standard deviation (sigma_h_wind) of wind speed at time interval h.
    """
    # Calculate k_w^h using equation (10)
    k_h_w = (sigma_h_wind / mu_h_wind) ** (-1.086)

    # Calculate c_w^h using equation (11)
    c_h_w = mu_h_wind / gamma(1 + 1 / k_h_w)

    return k_h_w, c_h_w

def calculate_f_wind(v_h, k_h_w, c_h_w):
    """
    Calculate the wind speed probability density function f_wind^h for a given wind speed v_h,
    mean (mu_h_wind), and standard deviation (sigma_h_wind) at time interval h.
    """

    # Implement the wind speed PDF using equation (9)
    f_wind_h = (k_h_w / c_h_w) * ((v_h / c_h_w) ** (k_h_w - 1)) * np.exp(-(v_h / c_h_w)) ** (k_h_w - 1)

    return f_wind_h

def calculate_chs(mu_solar, sigma_solar):
    """
    Calculate the parameter ch_s for the Beta distribution based on
    the mean (mu_solar) and standard deviation (sigma_solar) of solar irradiance.
    """
    if sigma_solar == 0 or mu_solar ==0:
        return 0
    numerator = (1 - mu_solar) * (((mu_solar * (1 + mu_solar)) / (sigma_solar**2)) - 1)
    return numerator

def calculate_khs(mu_solar, ch_s):
    """
    Calculate the parameter kh_s for the Beta distribution.
    """
    return (mu_solar / (1 - mu_solar)) * ch_s

def calculate_f_solar(sh, kh_s,ch_s):
    """
    Calculate the PDF of solar irradiance for a given solar irradiance value (sh),
    mean irradiance (mu_solar), and standard deviation (sigma_solar).
    """
    # Calculate ch_s and kh_s

    # Ensure kh_s and ch_s are greater than zero
    if kh_s <= 0 or ch_s <= 0:
        return 0

    # Calculate the Beta PDF using the equation (1)
    term1 = gamma(kh_s + ch_s) / (gamma(kh_s) * gamma(ch_s))
    pdf_value = term1 * (sh**(kh_s - 1)) * ((1 - sh)**(ch_s - 1))

    return pdf_value

def calculate_wind_power(v_h):
    if v_h < v_in or v_h > v_coff:
        return 0
    elif v_in <= v_h <= v_opt:
        return WIND_A * (v_h**3) + WIND_B * WTRATED
    elif v_opt <= v_h <= v_coff:
        return WTRATED
    else:
        return 0
    
def calculate_solar_power(s_h):
    Tc = Ta + s_h*((Tot-20)/0.8)
    Is = s_h*(Isc + Ki*(Tc-25))
    Vs = Voc - Kv*Tc
    return NSOLAR * FF * Vs * Is

def create_output_writer(net, time_steps, output_dir):
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xlsx", log_variables=list())
    ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_line', 'p_from_mw')
    ow.log_variable('res_line', 'p_to_mw')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_gen', 'p_mw')
    ow.log_variable('res_sgen', 'p_mw')
    return ow

def calculate_discomfort(xjh,pjh):
    """
    Calculate discomfort for all customers at once using vectorized operations.
    
    Parameters:
    - xjh: Array or list of curtailed power values for each customer
    - pjh: Array or list of demand values for each customer
    
    Returns:
    - Array of discomfort values for each customer
    """
    # Convert xjh to a NumPy array if it's not already
    xjh = np.array(xjh)
    pjh = np.array(pjh)
    
    # Calculate discomfort vectorized for all customers
    discomforts = np.exp(CONSUMER_BETA  * (xjh / pjh)) - 1
    return discomforts