import logging
#import os
from typing import Dict
import numpy as np
#import tensorflow as tf
import keras as keras
from constraints_rewards import *
from setting import *
from gym.spaces import Dict, Box, Discrete
from gym import spaces
from network_comp import *
from collections import deque
from pandapower.control.basic_controller import Controller

class MicrogridEnv(Controller):
    def __init__(self, w1, w2): 
        
        self.H = MAX_STEPS # Number of timesteps (planning horizon)
        self.J = NO_CONSUMERS  # Number of active consumers
        _,self.net = network_comp((0,0))
        self.ids = ids
        self.customer_ids = [f"C{i}" for i in range(32)]
        self.buffer = 5 # Number of previous timesteps to store for state
        super().__init__(self.net) 
        self.curtailment_indices = ["C8", "C21", "C13", "C29", "C24"]
        self.w1, self.w2 = w1, w2  # Weights for objectives
        self.prev_curtailed_buffer = deque(maxlen=self.buffer)
        self.prev_P_active_demand_buffer = deque(maxlen=self.buffer)

        self.action_space = spaces.Box(low=-1, high=1, shape=ACTION_SHAPE, dtype=np.float32)  #shape of action
        self.state = spaces.Box(low=-np.inf, high=np.inf, shape=STATE_SHAPE, dtype=np.float32)  #shape of state
        self.market_prices = price_profile_df.to_numpy()[:,1]
        self.original_datasource_consumers = load_profile_df
        self.datasource_consumers = self.original_datasource_consumers.copy()
        self.DfData_consumers = DFData(self.datasource_consumers)   
        self.applied = False

    def step(self, action,time):
        #action [(power_curtailed)x5, incentive rate]
        """
        Apply an action, update the environment state, and calculate rewards and penalties.
        """
        # Update state based on the action taken
        #assert self.action_space.contains(action), f"Action {action} is out of bounds!"
        scaled_action = scale_action(np.array(action)) # TODO scaled action wrong
        #assert np.all(scaled_action >= MIN_ACTION) and np.all(scaled_action <= MAX_ACTION), "Scaled action is out of bounds!"
        self.state,reward = self.update_state(self.state, scaled_action,time) 
        #assert self.observation_space.contains(self.state), f"State {self.state} is out of bounds!"
        _ = self.control_step(scaled_action,self.net, self.state[IDX_POWER_GEN])
        self.applied = False
        
        # TODO self.buffer.store, self.learn, if self training
        done = time >= self.H

        return self.state, reward, done, {} 
    
    def control_step(self, scaled_action,net, line_losses):
        if not self.applied:
            load_values = net.load.loc[self.curtailment_indices, 'p_mw'].to_numpy()
            updated_load_values = np.maximum(load_values - scaled_action[:-1], 0)
            net.load.loc[self.curtailment_indices, 'p_mw'] = updated_load_values
            net.gen.loc[self.ids.get('dg'), 'p_mw'] = line_losses
            self.applied = True
        else:
            print("misaligned control step")
    def _calculate_reward(self, scaled_action, state,time):
        curtailed = scaled_action[:-1]
        incentive = scaled_action[-1]   
        generation_cost = cal_costgen(state[IDX_POWER_GEN],state[IDX_PREV_GEN_COST]) 
        power_transfer_cost = cal_costpow(state[IDX_MARKET_PRICE], state[IDX_PGRID],state[IDX_PREV_POWER_TRANSFER_COST]) 
        mgo_profit = MGO_profit(state[IDX_MARKET_PRICE], curtailed, incentive,state[PREV_MGO_PROFIT])
        balance_penalty = power_balance_constraint(state[IDX_PGRID], state[IDX_POWER_GEN], state[IDX_SOLAR], 
                                                   state[IDX_WIND], state[IDX_TOTAL_LOAD], curtailed, 
                                                   state[IDX_LINE_LOSSES])  # TODO what is the point of this if its hardcoded?
        generation_penalty = generation_limit_constraint(state[IDX_POWER_GEN])
        ramp_penalty = ramp_rate_constraint(state[IDX_POWER_GEN], state[IDX_PREV_GENPOWER])
        curtailment_penalty = curtailment_limit_constraint(curtailed, state[IDX_ACTIVE_PMW]) 
        daily_curtailment_penalty = daily_curtailment_limit(curtailed, state[IDX_ACTIVE_PMW],    
                                                            state[IDX_PREV_CURTAILED], state[IDX_PREV_ACTIVE_PMW])
        consumer_incentives_penalty = indivdiual_consumer_benefit(incentive, curtailed, 
                                                                     state[IDX_DISCOMFORT], state[IDX_PREV_ACTIVE_BENEFIT]) 
        #incentives_limit_penalty = benefit_limit_constraint(incentive, curtailed, 
        #                                                       state[IDX_DISCOMFORT], state[PREV_ACTIVE_BENEFIT])
        incentive_rate_penalty = incentive_rate_constraint(incentive, state[IDX_MARKET_PRICE],state[IDX_MINMARKET_PRICE])
        budget_limit_penalty = budget_limit_constraint(incentive, curtailed, state[IDX_PREV_BUDGET])

        reward =  self.w1*(generation_cost + power_transfer_cost) - self.w2*mgo_profit - balance_penalty - generation_penalty - ramp_penalty - curtailment_penalty - daily_curtailment_penalty - consumer_incentives_penalty -  incentive_rate_penalty - budget_limit_penalty

        log_calc_rewards(t=time,
        source='Reward Calculation',
        freq=1, 
        penalties={
            "generation_cost": generation_cost,"power_transfer_cost": power_transfer_cost,"mgo_profit": mgo_profit,
            "balance_penalty": balance_penalty,"generation_penalty": generation_penalty,"ramp_penalty": ramp_penalty,
            "curtailment_penalty": curtailment_penalty,"daily_curtailment_penalty": daily_curtailment_penalty,
            "consumer_incentives_penalty": consumer_incentives_penalty,
            #"incentives_limit_penalty": incentives_limit_penalty,
            "incentive_rate_penalty": incentive_rate_penalty,"budget_limit_penalty": budget_limit_penalty,
        },
        reward=reward,
        scaled_action=scaled_action,
        state=state)

        return reward,generation_cost,power_transfer_cost,mgo_profit,consumer_incentives_penalty,budget_limit_penalty
    def update_state(self, state, action,time): 
        reward, generation_cost,power_transfer_cost,mgo_profit,consumer_incentives_penalty,budget_limit_penalty= self._calculate_reward(action, state,time)
        next_state = state.copy()

        # Fetch network response values based on the action
        line_losses, net = network_comp(TIMESTEPS = (time,time))

        curtailed = (action[:-1])
        P_demand = net.load.loc[self.customer_ids, 'p_mw'].to_numpy()
        P_demand_active = net.load.loc[self.curtailment_indices, 'p_mw'].to_numpy()
        pv_pw = net.sgen.at[ids.get('pv'), 'p_mw']
        wt_pw = net.gen.at[ids.get('wt'), 'p_mw']
        cdg_pw = net.gen.at[ids.get('dg'), 'p_mw']
        discomforts = calculate_discomfort(curtailed, P_demand_active) #only 5 customers!
        discomforts = np.clip(discomforts,-10000,10000)
        total_load = np.sum(P_demand)
        Pgrid = total_load - pv_pw - wt_pw - cdg_pw - np.sum(curtailed)
        market_price = self.market_prices[time]
        

        next_state[IDX_POWER_GEN] = cdg_pw
        next_state[IDX_PREV_GEN_COST] = generation_cost
        next_state[IDX_MARKET_PRICE] = market_price
        next_state[IDX_PGRID] = Pgrid
        next_state[IDX_PREV_POWER_TRANSFER_COST] = power_transfer_cost
        next_state[PREV_MGO_PROFIT] = mgo_profit
        next_state[IDX_SOLAR] = pv_pw
        next_state[IDX_WIND] = wt_pw
        next_state[IDX_TOTAL_LOAD] = total_load
        next_state[IDX_LINE_LOSSES] = line_losses
        next_state[IDX_PREV_GENPOWER] = state[IDX_POWER_GEN]
        next_state[IDX_ACTIVE_PMW] = P_demand_active
        next_state[IDX_PREV_CURTAILED] = self.prev_curtailed_buffer if len(self.prev_curtailed_buffer) > 1 else [0] * 5
        next_state[IDX_PREV_ACTIVE_PMW] =self.prev_P_active_demand_buffer if len(self.prev_P_active_demand_buffer) > 1 else [0] * 5
        next_state[IDX_DISCOMFORT] = discomforts
        next_state[IDX_PREV_ACTIVE_BENEFIT] = consumer_incentives_penalty
        next_state[IDX_MINMARKET_PRICE] = min(next_state[IDX_MARKET_PRICE],state[IDX_MINMARKET_PRICE])
        next_state[IDX_PREV_BUDGET] = budget_limit_penalty
        # Append current values to the tracking buffers
        self.prev_curtailed_buffer.append(curtailed)
        self.prev_P_active_demand_buffer.append(P_demand)

        return next_state, reward
    def reset(self):
        """
        Reset the environment to its initial state and return the initial observation.
        """
        line_losses, net = network_comp(TIMESTEPS=(0, 0))

        # Initialize state variables using network and profiles
        P_demand = net.load.loc[self.customer_ids, 'p_mw'].to_numpy()
        P_demand_active = net.load.loc[self.curtailment_indices, 'p_mw'].to_numpy()
        pv_pw = net.sgen.at[self.ids.get('pv'), 'p_mw']
        wt_pw = net.gen.at[self.ids.get('wt'), 'p_mw']
        cdg_pw = net.gen.at[self.ids.get('dg'), 'p_mw']
        market_price = self.market_prices[0]
        total_load = np.sum(P_demand)
        Pgrid = total_load - pv_pw - wt_pw - cdg_pw
        
        # Populate the state array
        self.state = np.array([0] * N_OBS)  # Reset state
        self.state[IDX_POWER_GEN] = cdg_pw
        self.state[IDX_MARKET_PRICE] = market_price
        self.state[IDX_PGRID] = Pgrid
        self.state[IDX_SOLAR] = pv_pw
        self.state[IDX_WIND] = wt_pw
        self.state[IDX_TOTAL_LOAD] = total_load
        self.state[IDX_LINE_LOSSES] = line_losses
        self.state[IDX_ACTIVE_PMW] = P_demand_active
        self.state[IDX_DISCOMFORT] = calculate_discomfort([0] * len(P_demand_active), P_demand_active)
        self.state[IDX_MINMARKET_PRICE] = market_price  # Initial min market price
        self.state[IDX_PREV_GEN_COST] = 0  # No previous generation cost initially
        self.state[IDX_PREV_POWER_TRANSFER_COST] = 0  # No previous power transfer cost
        self.state[IDX_PREV_ACTIVE_BENEFIT] = 0  # No previous benefit penalty
        self.state[IDX_PREV_BUDGET] = 0  # No previous budget penalty
        
        # Clear buffers for historical tracking
        self.prev_curtailed_buffer.clear()
        self.prev_P_active_demand_buffer.clear()
        self.prev_curtailed_buffer.append([0] * 5)
        self.prev_P_active_demand_buffer.append(P_demand_active)
        return self.state

    def close(self):
        """
        Perform any necessary cleanup.
        """
        pass  # Implement cleanup logic if needed