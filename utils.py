'''
func:
- normalize_df_column
- scale_to_mg
- normalize_state
- extra_reward
- plot_return
- plot_pf_results
- view_profile
- calculate_wind_shape
- calculate_f_wind
- calculate_chs
- calculate_khs
- beta_pdf_solar
- calculate_wind_power  
'''

import os
from pandapower.timeseries import OutputWriter
import logging
import pickle
from pathlib import Path
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from scipy.special import gamma
from setting import *

def normalize_df_column(df, column_name):
    col_min = df[column_name].min()
    col_max = df[column_name].max()
    return (df[column_name] - col_min) / (col_max - col_min)


# --- Action Scaling --- [-1,1] -> [min, max]
def scale_action(nn_action, min_action, max_action):
    nn_action = np.clip(nn_action, -1., 1.)
    return (nn_action + 1) * (max_action - min_action) / 2 + min_action

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

class NormalizeAction:
    def __init__(self, epsilon=1e-8):
        self.a_rms = RunningMeanStd(shape=(N_ACTION,))
        self.epsilon = epsilon

    def normalize(self, a):
        self.a_rms.update(a)
        a = (a -self.a_rms.mean) / np.sqrt(self.a_rms.var + self.epsilon)
        a = np.clip(a, -5, 5)
        return a

    def tf_normalize(self, a):
        mean = tf.convert_to_tensor(self.a_rms.mean, dtype=tf.float32)
        var = tf.convert_to_tensor(self.a_rms.var, dtype=tf.float32)
        a = (a - mean) / tf.math.sqrt(var + self.epsilon)
        a = tf.clip_by_value(a, -5, 5)
        return a

class NormalizeObservation:
    def __init__(self, epsilon=1e-8):
        self.obs_seq_rms = RunningMeanStd(shape=STATE_SEQ_SHAPE)
        self.obs_fnn_rms = RunningMeanStd(shape=STATE_FNN_SHAPE)
        self.epsilon = epsilon

    def normalize(self, obs, update=True):
        obs_seq, obs_fnn = obs
        if update:
            self.obs_seq_rms.update(obs_seq)
            self.obs_fnn_rms.update(obs_fnn)
        obs_seq = (obs_seq - self.obs_seq_rms.mean) / np.sqrt(self.obs_seq_rms.var + self.epsilon)
        obs_seq = np.clip(obs_seq, -5, 5)
        obs_fnn = (obs_fnn - self.obs_fnn_rms.mean) / np.sqrt(self.obs_fnn_rms.var + self.epsilon)
        obs_fnn = np.clip(obs_fnn, -5, 5)
        return obs_seq, obs_fnn
    
    def save(self, dir):
        fpath = Path(os.path.join(dir, 'obs.pkl'))
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(fpath, 'wb') as f:
            pickle.dump({
                'obs_seq_mean': self.obs_seq_rms.mean,
                'obs_seq_var': self.obs_seq_rms.var,
                'obs_fnn_mean': self.obs_fnn_rms.mean,
                'obs_fnn_var': self.obs_fnn_rms.var,
            }, f)

    def load(self, dir):
        with open(os.path.join(dir, 'obs.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.obs_seq_rms.mean = data['obs_seq_mean']
            self.obs_seq_rms.var = data['obs_seq_var']
            self.obs_fnn_rms.mean = data['obs_fnn_mean']
            self.obs_fnn_rms.var = data['obs_fnn_var']

class NormalizeReward:
    def __init__(self, gamma=GAMMA, epsilon=1e-8):
        self.return_rms = RunningMeanStd()
        self.return_ = np.zeros(1)
        self.gamma = gamma
        self.epsilon = epsilon

    def normalize(self, r):
        self.return_ = r + self.gamma * self.return_
        self.return_rms.update(self.return_)
        r /= np.sqrt(self.return_rms.var + self.epsilon)
        r = np.clip(r, -5, 5)
        return r

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


def normalize_state(state) -> Dict:
    normalized_state_rnn = state[0] / np.array([P_EXCESS_MAX, C_PRICE_MAX])
    # normalized_state_rnn = state[0] / np.array([*P_PV_MAX_LIST, *P_WT_MAX_LIST, *P_LOAD_MAX_LIST, C_PRICE_MAX])
    # normalized_state_rnn = state[0] / np.array([*P_PV_MAX_LIST, *P_WT_MAX_LIST, *P_LOAD_MAX_LIST, P_EXCESS_MAX, C_PRICE_MAX])
    normalized_state_fnn = state[1] / SOC_MAX

    return normalized_state_rnn, normalized_state_fnn


def extra_reward(nn_bat_p_mw, valid_bat_p_mw):
    # TODO look at paper to see what it does with penality for invalid action
    # penalty for invalid action
    dif = np.sum(np.abs(nn_bat_p_mw - valid_bat_p_mw))
    dif /= (P_B10_MAX + P_B5_MAX)
    reward = 0. if (dif < 1e-3) else (REWARD_INVALID_ACTION + dif * REWARD_INVALID_ACTION)
    return reward

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

def view_profile(pv_profile, wt_profile,load_profile,price_profile, start=None, length=None):
    start = 0 if start is None else start
    length = (len(pv_profile.index)-start) if length is None else length
    pv_p_mw = pv_profile.iloc[start: start+length, :]
    wt_p_mw = wt_profile.iloc[start: start+length, :]
    load_p_mw = load_profile.iloc[start: start+length, :]
    price_profile = price_profile.iloc[start: start+length, :]

    # MW and excess profile
    profile_p_mw = pd.concat([pv_p_mw, wt_p_mw, load_p_mw]).iloc[start: start+length, :]
    profile_p_mw = pd.concat([pv_p_mw, wt_p_mw]).iloc[start: start+length, :]
    excess_profile = pv_p_mw.sum(axis=1) + wt_p_mw.sum(axis=1) - load_p_mw.sum(axis=1)
    excess_profile = pd.DataFrame({'Excess': excess_profile})

    # info
    print('--- Profile ---')
    print(f'PV:\n max = {pv_profile.max(numeric_only=True)}, \nmin = {pv_profile.min(numeric_only=True)}')
    print(f'WT:\n max = {wt_profile.max(numeric_only=True)}, \nmin = {wt_profile.min(numeric_only=True)}')
    print(f'Load:\n max = {load_profile.max(numeric_only=True)}, \nmin = {load_profile.min(numeric_only=True)}')
    print(f'Excess:\n max = {excess_profile.max(numeric_only=True)}, \nmin = {excess_profile.min(numeric_only=True)}')
    print(f'Price:\n max = {price_profile.max(numeric_only=True)}, \nmin = {price_profile.min(numeric_only=True)}')

    # plot
    pv_p_mw.plot(xlabel='hour', ylabel='p_mw', title='PV')
    wt_p_mw.plot(xlabel='hour', ylabel='p_mw', title='WT')
    load_p_mw.plot(xlabel='hour', ylabel='p_mw', title='Load')
    plt.bar(TIMESTEPS, load_p_mw['C1'], label='C1', color='blue', bottom=None)
    plt.bar(TIMESTEPS, load_p_mw['C2'], label='C2', color='orange', bottom=load_p_mw['C1'])
    plt.bar(TIMESTEPS, load_p_mw['C3'], label='C3', color='purple', bottom=load_p_mw['C1'] + load_p_mw['C2'])
    plt.bar(TIMESTEPS, load_p_mw['C4'], label='C4', color='green', bottom=load_p_mw['C1'] + load_p_mw['C2'] + load_p_mw['C3'])
    plt.bar(TIMESTEPS, load_p_mw['C5'], label='C5', color='red', bottom=load_p_mw['C1'] + load_p_mw['C2'] + load_p_mw['C3'] + load_p_mw['C4'])
    price_profile.plot(xlabel='hour', ylabel='price', title='Price')
    profile_p_mw.plot(xlabel='hour', ylabel='p_mw', title='Microgrid')
    #ax = excess_profile.plot(xlabel='hour', ylabel='p_mw', title='excess')
    #ax.plot(range(start, start+length), np.zeros((length),))
    plt.show()
def plot_results2(filepath_results):
    # Load data from profiles


    # Calculate start and length for the data range
    start = 0  # or specify as needed
    length = len(pv_profile.index) - start

    # Extract profiles within the specified range
    pv_p_mw = pv_profile.iloc[start: start+length, :]
    wt_p_mw = wt_profile.iloc[start: start+length, :]
    load_p_mw = load_profile.iloc[start: start+length, :]
    price_profile = price_profile.iloc[start: start+length, :]

    # Calculate excess profile
    excess_profile = pv_p_mw.sum(axis=1) + wt_p_mw.sum(axis=1) - load_p_mw.sum(axis=1)
    excess_profile = pd.DataFrame({'Excess': excess_profile})

    # Display profile statistics
    print('--- Profile Statistics ---')
    print(f'PV Generation:\n Max = {pv_profile.max(numeric_only=True)}, Min = {pv_profile.min(numeric_only=True)}')
    print(f'Wind Generation:\n Max = {wt_profile.max(numeric_only=True)}, Min = {wt_profile.min(numeric_only=True)}')
    print(f'Load Demand:\n Max = {load_profile.max(numeric_only=True)}, Min = {load_profile.min(numeric_only=True)}')
    print(f'Excess Energy:\n Max = {excess_profile.max(numeric_only=True)}, Min = {excess_profile.min(numeric_only=True)}')
    print(f'Price Profile:\n Max = {price_profile.max(numeric_only=True)}, Min = {price_profile.min(numeric_only=True)}')

    # Plot PV profile
    pv_p_mw.plot(xlabel='Time Step', ylabel='Power (MW)', title='PV Power Generation (MW)')
    plt.show()

    # Plot Wind profile
    wt_p_mw.plot(xlabel='Time Step', ylabel='Power (MW)', title='Wind Power Generation (MW)')
    plt.show()

    # Plot Load profile
    load_p_mw.plot(xlabel='Time Step', ylabel='Power (MW)', title='Load Demand (MW)')
    plt.show()

    # Stacked bar plot for consumer loads
    plt.figure(figsize=(10, 6))
    plt.bar(load_p_mw.index, load_p_mw['C1'], label='C1', color='blue')
    plt.bar(load_p_mw.index, load_p_mw['C2'], bottom=load_p_mw['C1'], label='C2', color='orange')
    plt.bar(load_p_mw.index, load_p_mw['C3'], bottom=load_p_mw['C1'] + load_p_mw['C2'], label='C3', color='purple')
    plt.bar(load_p_mw.index, load_p_mw['C4'], bottom=load_p_mw['C1'] + load_p_mw['C2'] + load_p_mw['C3'], label='C4', color='green')
    plt.bar(load_p_mw.index, load_p_mw['C5'], bottom=load_p_mw['C1'] + load_p_mw['C2'] + load_p_mw['C3'] + load_p_mw['C4'], label='C5', color='red')
    plt.xlabel('Time Step')
    plt.ylabel('Power (MW)')
    plt.title('Stacked Load Demand per Consumer')
    plt.legend(loc="upper left")
    plt.show()

    # Plot Price profile
    price_profile.plot(xlabel='Time Step', ylabel='Price ($/MW)', title='Electricity Price Profile')
    plt.show()

    # Plot Microgrid power profile (combined PV and Wind generation)
    profile_p_mw = pd.concat([pv_p_mw, wt_p_mw], axis=1)
    profile_p_mw.plot(xlabel='Time Step', ylabel='Power (MW)', title='Microgrid Power Generation (PV and Wind)')
    plt.show()

    # Plot Excess Energy profile
    excess_profile.plot(xlabel='Time Step', ylabel='Excess Power (MW)', title='Excess Energy (Generation - Load)')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Add a zero line for reference
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


# must supply kwargs with net, ids, pcc_p_mw     
def log_cost_info(transaction_cost, t, source='', freq=100, **kwargs):
    """
    Logs cost and power flow information at specified time intervals with a source identifier.
    """
    if t % freq == 0:
        net = kwargs.get('net', None)
        ids = kwargs.get('ids', None)
        pcc_p_mw = kwargs.get('pcc_p_mw', None)

        if net is not None and ids is not None and pcc_p_mw is not None:
            p_wt = net.res_sgen.loc[ids['WT1'], 'p_mw'].sum() if 'WT1' in ids else 0
            p_pv = net.res_sgen.loc[ids['PV1'], 'p_mw'].sum() if 'PV1' in ids else 0
            p_cg = net.res_gen['p_mw'].sum()
            p_load = net.res_load['p_mw'].sum() if 'p_mw' in net.res_load else 0
            excess = p_pv + p_wt + p_cg - p_load

            logging.info(f'--- {source} Cost ---')
            logging.info(f'trans: {transaction_cost:.3f}')
            logging.info(f'--- Power flow from {source} ---')
            logging.info(f'pcc = {pcc_p_mw:.3f}, excess = {excess:.3f}, '
                         f'pv = {p_pv:.3f}, wt = {p_wt:.3f}, load = {p_load:.3f}')
        else:
            logging.warning("Missing data for logging in log_cost_info. Check 'kwargs' for required keys.")

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

def policy_simple(net, ids, bat5_soc, bat10_soc, bat5_max_e_mwh, bat10_max_e_mwh):
    # TODO simple, heuristic policy that balances excess power between charging / discharging batteries. LOOK AT PAPER
    p_pv = net.sgen.at[ids.get('pv3'), 'p_mw'] +\
        net.sgen.at[ids.get('pv4'), 'p_mw'] +\
        net.sgen.at[ids.get('pv5'), 'p_mw'] +\
        net.sgen.at[ids.get('pv6'), 'p_mw'] +\
        net.sgen.at[ids.get('pv8'), 'p_mw'] +\
        net.sgen.at[ids.get('pv9'), 'p_mw'] +\
        net.sgen.at[ids.get('pv10'), 'p_mw'] +\
        net.sgen.at[ids.get('pv11'), 'p_mw']
    p_wt = net.sgen.at[ids.get('wt7'), 'p_mw']
    p_load = net.load.at[ids.get('load_r1'), 'p_mw'] +\
        net.load.at[ids.get('load_r3'), 'p_mw'] +\
        net.load.at[ids.get('load_r4'), 'p_mw'] +\
        net.load.at[ids.get('load_r5'), 'p_mw'] +\
        net.load.at[ids.get('load_r6'), 'p_mw'] +\
        net.load.at[ids.get('load_r8'), 'p_mw'] +\
        net.load.at[ids.get('load_r10'), 'p_mw'] +\
        net.load.at[ids.get('load_r11'), 'p_mw']
                        
    p_b5_max = min((SOC_MAX - bat5_soc) * bat5_max_e_mwh / HOUR_PER_TIME_STEP, P_B5_MAX)
    p_b5_min = max((SOC_MIN - bat5_soc) * bat5_max_e_mwh / HOUR_PER_TIME_STEP, P_B5_MIN)
    p_b10_max = min((SOC_MAX - bat10_soc) * bat10_max_e_mwh / HOUR_PER_TIME_STEP, P_B10_MAX)
    p_b10_min = max((SOC_MIN - bat10_soc) * bat10_max_e_mwh / HOUR_PER_TIME_STEP, P_B10_MIN)

    excess = p_pv + p_wt - p_load
    # print(f'Excess = {excess}, pv: {p_pv}, wt: {p_wt}, load: {p_load}')
    if excess > 0:
        # charge
        b5_ratio = p_b5_max / (p_b5_max + p_b10_max) if (p_b5_max + p_b10_max) != 0. else 0.
        b10_ratio = p_b10_max / (p_b5_max + p_b10_max) if (p_b5_max + p_b10_max) != 0. else 0.
        p_b5 = min(excess * b5_ratio, p_b5_max)
        p_b10 = min(excess * b10_ratio, p_b10_max)
        # p_mgt5 = 0.
        # p_mgt9 = 0.
        # p_mgt10 = 0.
    else:
        # discharge
        b5_ratio = p_b5_min / (p_b5_min + p_b10_min) if (p_b5_min + p_b10_min) != 0. else 0.
        b10_ratio = p_b10_min / (p_b5_min + p_b10_min) if (p_b5_min + p_b10_min) != 0. else 0.
        p_b5 = max(excess * b5_ratio, p_b5_min)
        p_b10 = max(excess * b10_ratio, p_b10_min)
        p_b = p_b5 + p_b10

        # mgt5_ratio = P_MGT5_MAX / (P_MGT5_MAX + P_MGT9_MAX + P_MGT10_MAX)
        # mgt9_ratio = P_MGT9_MAX / (P_MGT5_MAX + P_MGT9_MAX + P_MGT10_MAX)
        # mgt10_ratio = P_MGT10_MAX / (P_MGT5_MAX + P_MGT9_MAX + P_MGT10_MAX)
        # mgt5_op_point = (C_BUY - C_MGT5[1]) / C_MGT5[0]
        # mgt9_op_point = (C_BUY - C_MGT9[1]) / C_MGT9[0]
        # mgt10_op_point = (C_BUY - C_MGT10[1]) / C_MGT10[0]
        # p_mgt5 = 0. if excess > p_b  else min((p_b - excess) * mgt5_ratio, mgt5_op_point)
        # p_mgt9 = 0. if excess > p_b  else min((p_b - excess) * mgt9_ratio, mgt9_op_point)
        # p_mgt10 = 0. if excess > p_b  else min((p_b - excess) * mgt10_ratio, mgt10_op_point)
    
    return np.array([p_b5, p_b10])

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
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_gen', 'p_mw')
    ow.log_variable('res_sgen', 'p_mw')
    return ow
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