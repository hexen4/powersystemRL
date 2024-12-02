import logging
#import os
from typing import Dict
import numpy as np
#import tensorflow as tf
import keras as keras
from constraints_rewards import *
from setting import *
from gym import spaces
from network_comp import network_comp
from collections import deque
class MicrogridEnv():
    def __init__(self, net, ids, initial_state, w1, w2,H = len(TIMESTEPS), J = NO_CONSUMERS, buffer = 5):  
        self.H = H  # Number of timesteps (planning horizon)
        self.J = J  # Number of active consumers
        self.ids = ids
        self.current_timestep = 0

        self.w1, self.w2 = w1, w2  # Weights for objectives
        self.net = net
        self.prev_curtailed_buffer = deque(maxlen=self.buffer)
        self.prev_P_demand_buffer = deque(maxlen=self.buffer)
        self.prev_discomforts_buffer = deque(maxlen=self.buffer)
        self.prev_genpower_buffer = deque(maxlen=self.buffer)
        self.reward_history = []
        self.buffer = buffer

        self.initial_state = initial_state 
        self.action_space = spaces.Box(low=-1, high=1, shape=(N_ACTION,1), dtype=np.float32)  #shape of action
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(N_INTERMITTENT_STATES+N_CONTROLLABLE_STATES,1), dtype=np.float32)  # shape of  state
        self.state_seq = initial_state[:N_INTERMITTENT_STATES+1]
        self.state_fnn = initial_state[N_INTERMITTENT_STATES:]
        self.market_prices = price_profile_df.to_numpy()
        self.original_datasource_consumers = load_profile_df
        self.datasource_consumers = self.original_datasource_consumers.copy()
        self.DfData_consumers = DFData(self.datasource_consumers)
    
        
    def step(self, action):
        #action [(power_curtailed)x5, incentive rate]
        """
        Apply an action, update the environment state, and calculate rewards and penalties.
        """
        # Update state based on the action taken
        assert self.action_space.contains(action), f"Action {action} is out of bounds!"
        scaled_action = scale_action(action)  

        self.state_seq,self.state_fnn = self.update_state(scaled_action) #need to do this for st+1 ->. IC at t = 0
        assert self.observation_space.contains([self.state_seq,self.state_fnn]), f"State {self.state} is out of bounds!"
    
        reward = self._calculate_reward(action, self.state) 
        self.reward_history.append(reward)
        self.last_time_step = self.current_timestep
        self.current_timestep += 1
        self.applied = False
        
        # TODO self.buffer.store, self.learn, if self training
        done = self.current_timestep >= self.H

        return self.state, reward, done, {} # TODO make sure reward cumultative in next func.
    
    def _calculate_reward(self, scaled_action, state):

        generation_cost = cal_costgen(state[IDX_POWER_GEN]) 
        power_transfer_cost = cal_costpow(state[IDX_MARKET_PRICE], state[IDX_PGRID]) 
        mgo_profit = MGO_profit(state[IDX_MARKET_PRICE], scaled_action[:-1], scaled_action[-1])
        balance_penalty = power_balance_constraint(state[IDX_PGRID], state[IDX_POWER_GEN], state[IDX_SOLAR], 
                                                   state[IDX_WIND], state[IDX_CUSTOMER_PMW], scaled_action[:-1], 
                                                   state[IDX_LINE_LOSSES]) 
        generation_penalty = generation_limit_constraint(state[IDX_POWER_GEN])
        ramp_penalty = ramp_rate_constraint(state[IDX_POWER_GEN], state[IDX_PREV_GENPOWER])
        curtailment_penalty = curtailment_limit_constraint(scaled_action[:-1], state[IDX_CUSTOMER_PMW])
        daily_curtailment_penalty = daily_curtailment_limit(scaled_action[:-1], state[IDX_CUSTOMER_PMW], 
                                                            state[IDX_PREV_CURTAILED], state[IDX_PREV_DEMAND])
        consumer_incentives_penalty = consumer_incentives_constraint(scaled_action[-1], scaled_action[:-1], 
                                                                     state[IDX_DISCOMFORT])
        incentives_limit_penalty = incentives_limit_constraint(scaled_action[-1], scaled_action[:-1], 
                                                               state[IDX_DISCOMFORT], state[IDX_PREV_CURTAILED], 
                                                               state[IDX_PREV_DISCOMFORT])
        incentive_rate_penalty = incentive_rate_constraint(scaled_action[-1], state[IDX_MARKET_PRICE])
        budget_limit_penalty = budget_limit_constraint(scaled_action[-1], scaled_action[:-1])

        reward =  self.w1*(generation_cost + power_transfer_cost) - self.w2*mgo_profit - balance_penalty - generation_penalty - ramp_penalty - curtailment_penalty - daily_curtailment_penalty - consumer_incentives_penalty - incentives_limit_penalty - incentive_rate_penalty - budget_limit_penalty
        
        log_calc_rewards(t=self.current_timestep,
        source='Reward Calculation',
        freq=5,  # Change frequency to control logging intervals
        penalties={
            "generation_cost": generation_cost,"power_transfer_cost": power_transfer_cost,"mgo_profit": mgo_profit,
            "balance_penalty": balance_penalty,"generation_penalty": generation_penalty,"ramp_penalty": ramp_penalty,
            "curtailment_penalty": curtailment_penalty,"daily_curtailment_penalty": daily_curtailment_penalty,
            "consumer_incentives_penalty": consumer_incentives_penalty,"incentives_limit_penalty": incentives_limit_penalty,
            "incentive_rate_penalty": incentive_rate_penalty,"budget_limit_penalty": budget_limit_penalty,
        },
        reward=reward,
        scaled_action=scaled_action,
        state=state)
        return reward
    def update_state(self, action): # TODO sort this out. 
        state_seq = [0] * len(self.state_seq_shape[1])
        state_fnn = [0] * len(self.state_fnn_shape)  


        # Fetch network response values based on the action
        line_losses, net, ids = network_comp(TIMESTEPS = self.current_timestep)

        curtailed = np.array((action[:-1]))
        P_demand = net.load.loc[self.customer_ids, 'p_mw'].to_numpy()
        pv_pw = net.sgen.at[ids.get('pv'), 'p_mw']
        wt_pw = net.gen.at[ids.get('wt'), 'p_mw']
        cdg_pw = net.gen.at[ids.get('dg'), 'p_mw']
        discomforts = calculate_discomfort(curtailed, P_demand).to_numpy()
        Pgrid = np.sum(P_demand) - pv_pw - wt_pw - cdg_pw - np.sum(curtailed)
        market_price = self.market_prices[self.current_timestep]
        

        state_seq[IDX_SOLAR] = pv_pw  # Solar generation
        state_seq[IDX_WIND] = wt_pw   # Wind generation
        state_seq[IDX_CUSTOMER_PMW] = P_demand  # Customer power demand
        state_seq[IDX_MARKET_PRICE] = market_price  # Market price
        state_seq[IDX_LINE_LOSSES] = line_losses  # Line losses


        state_fnn[IDX_POWER_GEN] = cdg_pw
        state_fnn[IDX_PGRID] = Pgrid
        state_fnn[IDX_DISCOMFORT] = discomforts
        state_fnn[IDX_PREV_GENPOWER] = self.prev_genpower_buffer[-1] if len(self.prev_genpower_buffer) > 0 else 0
        state_fnn[IDX_PREV_CURTAILED] = self.prev_curtailed_buffer[-1] if len(self.prev_curtailed_buffer) > 0 else [0] * 5
        state_fnn[IDX_PREV_DEMAND] = self.prev_P_demand_buffer[-1] if len(self.prev_P_demand_buffer) > 0 else [0] * 5
        state_fnn[IDX_PREV_DISCOMFORT] = self.prev_discomforts_buffer[-1] if len(self.prev_discomforts_buffer) > 0 else [0] * 5

        # Append current values to the tracking buffers
        self.prev_curtailed_buffer.append(curtailed)
        self.prev_P_demand_buffer.append(P_demand)
        self.prev_discomforts_buffer.append(discomforts)
        self.prev_genpower_buffer.append(cdg_pw)

        return state_seq, state_fnn
    def reset(self):
        """
        Reset the environment to its initial state and return the initial observation.
        """
        self.state = self.initial_state
        self.current_timestep = 0
        self.prev_genpower_buffer.clear()
        self.prev_curtailed_buffer.clear()
        self.prev_P_demand_buffer.clear()
        self.prev_discomforts_buffer.clear()
        self.reward_history = []
        self.datasource_consumers = self.original_datasource_consumers.copy()
        self.data_source_consumers = DFData(self.datasource_consumers)
        self.applied = False
        return self.state

    def close(self):
        """
        Perform any necessary cleanup.
        """
        pass  # Implement cleanup logic if needed