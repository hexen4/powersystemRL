import logging
#import os
from typing import Dict
import numpy as np
#import tensorflow as tf
import keras as keras
from constraints_rewards import *
#from controllers.models import ActorMuModel, CriticQModel, SequenceModel, get_mu_actor, get_q_critic
#from controllers.buffer import ReplayBuffer, PrioritizedReplayBuffer
from setting import *
from gym import spaces
from network_comp import network_comp

    #action [(power_curtailed)x5, incentive rate]
class MicrogridEnv:
    def __init__(self, net, ids, initial_state, w1, w2,H = len(TIMESTEPS), J = NO_CONSUMERS, buffer = 5): #what should i put here?   
        super(MicrogridEnv, self).__init__()
        self.H = H  # Number of timesteps (planning horizon)
        self.J = J  # Number of consumers
        self.current_timestep = 0
        self.w1, self.w2 = w1, w2  # Weights for objectives
        self.net = net
        self.ids = ids

        self.reward_history = []  
        self.prev_genpower_buffer = []
        self.prev_curtailed_buffer = []
        self.prev_P_demand_buffer = []
        self.prev_discomforts_buffer = []
        self.buffer = buffer

        self.initial_state = initial_state # TODO initial conditions at t = 0, same format as next_state 
        self.action_space = spaces.Box(low=-1, high=1, shape=N_ACTION, dtype=np.float32)  #shape of action
        self.N_OBS = IDX_PREV_DISCOMFORT + 1  # Observation space dimension
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.N_OBS,1), dtype=np.float32)  # shape of  state
        
    def step(self, action):
        """
        Apply an action, update the environment state, and calculate rewards and penalties.
        """
        self.current_timestep += 1
        # Update state based on the action taken
        assert self.action_space.contains(action), f"Action {action} is out of bounds!"
        scaled_action = scale_action(action)  

        self.state = self.update_state(scaled_action) #need to do this for st+1 ->. IC at t = 0
        assert self.observation_space.contains(self.state), f"State {self.state} is out of bounds!"

        reward = self._calculate_reward(action, self.state) 
        self.reward_history.append(reward)
        done = self.current_timestep >= self.H

        return self.state, reward, done, {} # TODO make sure reward cumultative in next func.
    
# action -> grid values -> state (what should be in state)> reward -> next state
    def _calculate_reward(self, scaled_action, state):

        generation_cost = cal_costgen(state[IDX_POWER_GEN]) #from calc_grid_values
        power_transfer_cost = cal_costpow(state[IDX_MARKET_PRICE], state[IDX_PGRID]) #from data + #from calc_grid_values
        mgo_profit = MGO_profit(state[IDX_MARKET_PRICE], scaled_action[:-1], scaled_action[-1]) # action space, data
        balance_penalty = power_balance_constraint(state[IDX_PGRID], state[IDX_POWER_GEN], 
                                                    state[IDX_SOLAR], state[IDX_WIND], state[IDX_CUSTOMER_PMW], scaled_action[:-1], P_loss) 
            # TODO p_loss
        generation_penalty = generation_limit_constraint(state[IDX_POWER_GEN])
        ramp_penalty = ramp_rate_constraint(state[IDX_POWER_GEN], state[IDX_PREV_POWER])
        curtailment_penalty = curtailment_limit_constraint(scaled_action[:-1], state[IDX_CUSTOMER_PMW])
        daily_curtailment_penalty = daily_curtailment_limit(scaled_action[:-1], state[IDX_CUSTOMER_PMW], state[IDX_PREV_CURTAILED], state[IDX_PREV_DEMAND])
        consumer_incentives_penalty = consumer_incentives_constraint(scaled_action[-1], scaled_action[:-1], state[IDX_DISCOMFORT])
        incentives_limit_penalty = incentives_limit_constraint(scaled_action[-1], scaled_action[:-1], state[IDX_DISCOMFORT], state[IDX_PREV_CURTAILED], state[IDX_PREV_DISCOMFORT])
        incentive_rate_penalty = incentive_rate_constraint(scaled_action[-1], state[IDX_MARKET_PRICE])
        budget_limit_penalty = budget_limit_constraint(scaled_action[-1], scaled_action[:-1])

        return self.w1*(generation_cost + power_transfer_cost) - self.w2*mgo_profit - balance_penalty - generation_penalty - ramp_penalty - curtailment_penalty - daily_curtailment_penalty - consumer_incentives_penalty - incentives_limit_penalty - incentive_rate_penalty - budget_limit_penalty
        
    def update_state(self, action):
        next_state = [0] * self.N_OBS

        # Fetch network response values based on the action
        res, net, ids = network_comp(action)

        curtailed = list(action[:-1])
        P_demand = [net.load.at[ids.get(f'c{i}'), 'p_mw'] for i in range(1, 6)]
        pv_pw = net.sgen.at[ids.get('pv'), 'p_mw']
        wt_pw = net.gen.at[ids.get('wt'), 'p_mw']
        cdg_pw = net.gen.at[ids.get('dg'), 'p_mw']
        discomforts = calculate_discomfort(curtailed, P_demand)

        # Append current values to the tracking buffers
        self.prev_curtailed_buffer.append(curtailed)
        self.prev_P_demand_buffer.append(P_demand)
        self.prev_discomforts_buffer.append(discomforts)

        # Ensure the tracking buffers only hold the most recent `buffer` values
        if len(self.prev_curtailed_buffer) > self.buffer:
            self.prev_curtailed_buffer.pop(0)
        if len(self.prev_P_demand_buffer) > self.buffer:
            self.prev_P_demand_buffer.pop(0)
        if len(self.prev_discomforts_buffer) > self.buffer:
            self.prev_discomforts_buffer.pop(0)
        # Populate `next_state` with power generation and customer demand values
        next_state[IDX_POWER_GEN] = cdg_pw
        next_state[IDX_SOLAR] = pv_pw
        next_state[IDX_WIND] = wt_pw
        next_state[IDX_CUSTOMER_PMW] = P_demand

        # Calculate Pgrid and assign it to the state
        Pgrid = sum(P_demand) - pv_pw - wt_pw - cdg_pw - np.sum(curtailed)
        next_state[IDX_PGRID] = Pgrid

        # Fetch market price and discomfort cost, and add them to `next_state`
        market_price = price_profile_df.iloc[self.current_timestep]
        next_state[IDX_MARKET_PRICE] = market_price
        next_state[IDX_DISCOMFORT] = discomforts

        self.prev_genpower_buffer.append(next_state[IDX_POWER_GEN])
        if len(self.prev_genpower_buffer) > self.buffer:
            self.prev_genpower_buffer.pop(0)

        # Assign previous values or defaults to `next_state`
        next_state[IDX_PREV_POWER] = self.prev_genpower_buffer[-2] if len(self.prev_genpower_buffer) > 1 else 0
        next_state[IDX_PREV_CURTAILED] = self.prev_curtailed_buffer[-2] if len(self.prev_curtailed_buffer) > 1 else [0] * 5
        next_state[IDX_PREV_DEMAND] = self.prev_P_demand_buffer[-2] if len(self.prev_P_demand_buffer) > 1 else [0] * 5
        next_state[IDX_PREV_DISCOMFORT] = self.prev_discomforts_buffer[-2] if len(self.prev_discomforts_buffer) > 1 else [0] * 5


        #[P_gen, P_solar, P_wind, P_demand, P_grid, market_price, discomfort, prev_genpower, prev_curtailed, prev_P_demand, prev_discomforts]
        return next_state
    def reset(self):
        """
        Reset the environment to its initial state and return the initial observation.
        """
        self.state = self.initial_state
        self.current_timestep = 0
        self.reward_history = []
        self.prev_genpower_buffer = []
        self.prev_curtailed_buffer = []
        self.prev_P_demand_buffer = []
        self.prev_discomforts_buffer = []
        return self.state

    def close(self):
        """
        Perform any necessary cleanup.
        """
        pass  # Implement cleanup logic if needed