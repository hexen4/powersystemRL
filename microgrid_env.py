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


class MicrogridEnv():
    def __init__(self, w1, w2): 
        
        self.H = MAX_STEPS - 1 # Number of timesteps (planning horizon)
        self.ids = ids
        self.customer_ids = [f"C{i}" for i in range(32)] #indexed C0...
        self.curtailment_indices = ["C8", "C21", "C13", "C29", "C24"] #correspond to C1..C5 in paper
        self.w1, self.w2 = w1, w2  # Weights for objectives

        self.action_space = spaces.Box(low=-1, high=1, shape=ACTION_SHAPE, dtype=np.float32)  #shape of action
        self.state = spaces.Box(low=-np.inf, high=np.inf, shape=STATE_SHAPE, dtype=np.float32)  #shape of state
        self.market_prices = price_profile_df.to_numpy()[:,1]
        self.original_datasource_consumers = load_profile_df

         #TODO wrong, need to fix NR to follow papers
        self.init_line_losses, self.init_net = network_comp((0, 0), scaled_action=[0] * 6)
        # Initialize variables using self.init_net
        self.init_P_demand = self.init_net.load.loc[self.customer_ids, 'p_mw'].to_numpy()
        self.init_P_demand_active = self.init_net.load.loc[self.curtailment_indices, 'p_mw'].to_numpy()
        self.init_pv_pw = self.init_net.sgen.at[self.ids.get('pv'), 'p_mw']
        self.init_wt_pw = self.init_net.gen.at[self.ids.get('wt'), 'p_mw']
        self.init_cdg_pw = self.init_net.gen.at[self.ids.get('dg'), 'p_mw']
        self.init_market_price = self.market_prices[0]
        self.init_total_load = np.sum(self.init_P_demand) 

    def step(self, state, action,time):
        #action [(power_curtailed)x5, incentive rate]
        """
        Apply an action, calculate rewards and penalties, update state if not at the last step.
        """
        #scale action
        max_incentive = self.market_prices[time]
        max_action = np.append(state[IDX_ACTIVE_PMW], max_incentive) #TODO does it make sense for max_action to be the market price
        scaled_action = scale_action(np.array(action),max_action)  #at
        
        #calculate reward for st
        reward, generation_cost,power_transfer_cost,mgo_profit,prev_benefit,prev_budget= self.calculate_reward(scaled_action,state,time) #rt
        #logging s_t, a_t, r_t
        
        #update state stopping at t = 23
        self.state = self.update_state(state, scaled_action,time, generation_cost,power_transfer_cost,
                                    mgo_profit,prev_benefit,prev_budget)  #s_t+1

        done = time >= self.H
        return self.state, reward, done, {} 
    
    def update_state(self, state, action,time,generation_cost,power_transfer_cost,mgo_profit,prev_benefit,prev_budget): 
        #calculate params for st+1
        curtailed = action[:-1]
        time += 1
        line_losses, self.net = network_comp(TIMESTEPS = (time,time),scaled_action = action)
        pv_pw = self.net.sgen.at[ids.get('pv'), 'p_mw']
        wt_pw = self.net.gen.at[ids.get('wt'), 'p_mw']
        cdg_pw = self.net.gen.at[ids.get('dg'), 'p_mw']
        market_price = self.market_prices[time]
        load_before_curtail = self.original_datasource_consumers.iloc[time]
        P_demand_active = load_before_curtail.loc[self.curtailment_indices]
        total_load = np.sum(load_before_curtail)


        #populate st+1
        next_state = np.array([0] * N_OBS).astype(np.float32)
        next_state[IDX_POWER_GEN] = np.float32(cdg_pw)
        next_state[IDX_PREV_GEN_COST] = np.float32(generation_cost)
        next_state[IDX_MARKET_PRICE] = np.float32(market_price)
        next_state[IDX_PREV_POWER_TRANSFER_COST] = np.float32(power_transfer_cost)
        next_state[PREV_MGO_PROFIT] = np.float32(mgo_profit)
        next_state[IDX_SOLAR] = np.float32(pv_pw)
        next_state[IDX_WIND] = np.float32(wt_pw)
        next_state[IDX_TOTAL_LOAD] = np.float32(total_load)
        next_state[IDX_LINE_LOSSES] = np.float32(line_losses)
        next_state[IDX_PREV_GENPOWER] = state[IDX_POWER_GEN]
        next_state[IDX_ACTIVE_PMW] = np.array(P_demand_active, dtype=np.float32)
        next_state[IDX_PREV_CURTAILED] = np.array(curtailed, dtype=np.float32)
        next_state[IDX_PREV_ACTIVE_PMW] = state[IDX_ACTIVE_PMW]
        next_state[IDX_PREV_ACTIVE_BENEFIT] = np.float32(prev_benefit)
        next_state[IDX_MINMARKET_PRICE] = np.float32(min(next_state[IDX_MARKET_PRICE], state[IDX_MINMARKET_PRICE]))
        next_state[IDX_PREV_BUDGET] = np.float32(prev_budget)
 


        return next_state
    
    def calculate_reward(self, scaled_action, state,time):
        curtailed = scaled_action[:-1]
        incentive = scaled_action[-1]     
        P_demand_active = state[IDX_ACTIVE_PMW]
        P_grid = state[IDX_TOTAL_LOAD] - state[IDX_WIND]- state[IDX_SOLAR] - np.sum(curtailed)
        discomforts = calculate_discomfort(curtailed, P_demand_active) #only 5 customers!
         
        generation_cost = cal_costgen(state[IDX_POWER_GEN],state[IDX_PREV_GEN_COST]) 
        power_transfer_cost = cal_costpow(state[IDX_MARKET_PRICE],P_grid,state[IDX_PREV_POWER_TRANSFER_COST]) 
        mgo_profit = MGO_profit(state[IDX_MARKET_PRICE], curtailed, incentive,state[PREV_MGO_PROFIT])
        balance_penalty = power_balance_constraint(P_grid, state[IDX_POWER_GEN], state[IDX_SOLAR], 
                                                   state[IDX_WIND], state[IDX_TOTAL_LOAD], curtailed, 
                                                   state[IDX_LINE_LOSSES])  # TODO what is the point of this if its hardcoded?
        generation_penalty = generation_limit_constraint(state[IDX_POWER_GEN])
        ramp_penalty = ramp_rate_constraint(state[IDX_POWER_GEN], state[IDX_PREV_GENPOWER],time)
        curtailment_penalty = curtailment_limit_constraint(curtailed, state[IDX_ACTIVE_PMW]) 
        daily_curtailment_penalty = daily_curtailment_limit(curtailed, state[IDX_ACTIVE_PMW],    
                                                            state[IDX_PREV_CURTAILED], state[IDX_PREV_ACTIVE_PMW])
        consumer_benefit_penalty,prev_benefit = indivdiual_consumer_benefit(incentive, curtailed, 
                                                                     discomforts, state[IDX_PREV_ACTIVE_BENEFIT]) 
        #incentives_limit_penalty = benefit_limit_constraint(incentive, curtailed, 
        #                                                       state[IDX_DISCOMFORT], state[PREV_ACTIVE_BENEFIT])
        incentive_rate_penalty = incentive_rate_constraint(incentive, state[IDX_MARKET_PRICE],state[IDX_MINMARKET_PRICE])
        budget_limit_penalty, prev_budget = budget_limit_constraint(incentive, curtailed, state[IDX_PREV_BUDGET])

        reward =  - self.w1*(generation_cost + power_transfer_cost) + self.w2*mgo_profit - balance_penalty - generation_penalty - ramp_penalty - curtailment_penalty - daily_curtailment_penalty - consumer_benefit_penalty -  incentive_rate_penalty - budget_limit_penalty
        
        log_calc_rewards(t=time,
        source='Reward Calculation',
        freq=1, 
        penalties={
            "generation_cost": generation_cost,"power_transfer_cost": power_transfer_cost,"mgo_profit": mgo_profit,
            "balance_penalty": balance_penalty,"generation_penalty": generation_penalty,"ramp_penalty": ramp_penalty,
            "curtailment_penalty": curtailment_penalty,"daily_curtailment_penalty": daily_curtailment_penalty,
            "consumer_benefit_penalty": consumer_benefit_penalty,
            #"incentives_limit_penalty": incentives_limit_penalty,
            "incentive_rate_penalty": incentive_rate_penalty,"budget_limit_penalty": budget_limit_penalty,
        },
        reward=reward,
        scaled_action=scaled_action,
        state=state)

        return reward.astype(np.float32),generation_cost,power_transfer_cost,mgo_profit,prev_benefit,prev_budget
    def reset(self):
        """
        Reset the environment to its initial state and return the initial observation.
        """
        self.net = self.init_net #reset net
        state = np.array([0] * N_OBS).astype(np.float32)
        
        # Populate the state array
        state[IDX_POWER_GEN] = np.float32(self.init_cdg_pw)
        state[IDX_MARKET_PRICE] = np.float32(self.init_market_price)
        state[IDX_SOLAR] = np.float32(self.init_pv_pw)
        state[IDX_WIND] = np.float32(self.init_wt_pw)
        state[IDX_TOTAL_LOAD] = np.float32(self.init_total_load)
        state[IDX_LINE_LOSSES] = np.float32(self.init_line_losses)
        state[IDX_ACTIVE_PMW] = np.array(self.init_P_demand_active, dtype=np.float32)
        state[IDX_MINMARKET_PRICE] = self.init_market_price 
        state[IDX_PREV_GEN_COST] = 0  
        state[IDX_PREV_POWER_TRANSFER_COST] = 0  
        state[IDX_PREV_ACTIVE_BENEFIT] = 0  
        state[IDX_PREV_BUDGET] = 0 
        

        return state

