import logging
import os
from typing import Dict
import numpy as np
import tensorflow as tf
import keras as keras
from constraints_rewards import *
#from tensorflow.keras.optimizers import Adam for testing need later
from pandapower.control.basic_controller import Controller
from controllers.models import ActorMuModel, CriticQModel, SequenceModel, get_mu_actor, get_q_critic
from controllers.buffer import ReplayBuffer, PrioritizedReplayBuffer
from setting import *
from gym import spaces
from network_comp import network_comp

#action [(power_curtailed), incentive rate]
class MicrogridEnv:
    def __init__(self, net, ids, initial_state, incentive, w1, w2,H = len(TIMESTEPS), J = NO_CONSUMERS):
        super(MicrogridEnv, self).__init__()
        self.H = H  # Number of timesteps (planning horizon)
        self.J = J  # Number of consumers
        self.initial_state = initial_state
        self.incentive = incentive  # incentive rate
        self.w1, self.w2 = w1, w2  # Weights for objectives
        self.net = net
        self.ids = ids
        # Define action and observation space
        # Adjust dimensions as needed for the problem specifics
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # Example action space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(H + J + 2,), dtype=np.float32)  # Example observation space
        # Initialize state
        self.state = self.initial_state
        self.current_timestep = 0
    def step(self, action):
        """
        Apply an action, update the environment state, and calculate rewards and penalties.
        """
        # Update state based on the action taken
        next_state = self._update_state(action)

        # Calculate the reward for the current step
        reward = self._calculate_reward(action, next_state) # TODO check if next_state is correct

        # Log the total reward for history tracking
        self.reward_history.append(reward)

        # Check if the episode has ended (e.g., reached the end of the time horizon)
        done = self.current_timestep >= self.H

        # Update the current state and timestep counter
        self.state = next_state
        self.current_timestep += 1

        return next_state, reward, done, {}
# action -> grid values -> state (what should be in state)> reward -> next state
    def _calculate_reward(self, action, state):
        # Implement reward calculation logic or call an external function
        generation_cost = cal_costgen(power_gen) #from calc_grid_values
        power_transfer_cost = cal_costpow(alpha, power_transfer) #from data + #from calc_grid_values
        mgo_profit = MGO_profit(alpha, curtailed, incentive) # action space, data
        balance_penalty = power_balance_constraint(P_grid, P_gen, P_solar, P_wind, P_demand, curtailed, P_loss)
        generation_penalty = generation_limit_constraint(P_gen)
        ramp_penalty = ramp_rate_constraint(P_gen, P_gen_prev)
        curtailment_penalty = curtailment_limit_constraint(curtailed, P_demand)
        daily_curtailment_penalty = daily_curtailment_limit(curtailed, P_demand, prev_curtailed, prev_P_demand)
        consumer_incentives_penalty = consumer_incentives_constraint(incentive, curtailed, discomforts)
        incentives_limit_penalty = incentives_limit_constraint(incentive, curtailed, discomforts, prev_curtailed, prev_discomforts)
        incentive_rate_penalty = incentive_rate_constraint(incentive, alpha)
        budget_limit_penalty = budget_limit_constraint(incentive, curtailed, budget)
        log_cost_info(mgo_profit, self.current_timestep, kwargs=self.net, source='MGO_profit')    
        return self.w1*(generation_cost + power_transfer_cost) - self.w2*mgo_profit - balance_penalty - generation_penalty - ramp_penalty - curtailment_penalty - daily_curtailment_penalty - consumer_incentives_penalty - incentives_limit_penalty - incentive_rate_penalty - budget_limit_penalty
        
    def _update_state(self, action):
        # Implement state transition logic
        return new_state    

    
    def _calculate_grid_values(self, action):
        net, ids = network_comp(action) 
        power_gen = net.res_bus["vm_pu"].values # TODO check this
        return power_gen, net, ids
    def reset(self):
        """
        Reset the environment to its initial state and return the initial observation.
        """
        self.state = self.initial_state
        self.current_timestep = 0
        self.reward_history = []  # Reset reward history if needed
        return self.state

    def close(self):
        """
        Perform any necessary cleanup.
        """
        pass  # Implement cleanup logic if needed