import logging
import os
from typing import Dict
import numpy as np
import tensorflow as tf
import keras as keras
#from tensorflow.keras.optimizers import Adam for testing need later
from pandapower.control.basic_controller import Controller
from controllers.models import ActorMuModel, CriticQModel, SequenceModel, get_mu_actor, get_q_critic
from controllers.buffer import ReplayBuffer, PrioritizedReplayBuffer
from setting import *
from gym import spaces

class MicrogridEnv:
    def __init__(self, H, J, initial_state, alpha, r, a1, a2, a3, w1, w2):
        super(MicrogridEnv, self).__init__()
        self.H = H  # Number of timesteps (planning horizon)
        self.J = J  # Number of consumers
        self.initial_state = initial_state
        self.alpha = alpha  # Coefficients for cost functions
        self.r = r  # Curtailment compensation rates
        self.a1, self.a2, self.a3 = a1, a2, a3  # Coefficients for fuel cost
        self.w1, self.w2 = w1, w2  # Weights for objectives

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
        reward = self._calculate_reward(action, next_state)

        # Calculate constraint penalties (if applicable)
        constraint_penalty = self._calculate_constraints(action, next_state)

        # Combine reward and penalties to get the total reward
        total_reward = reward - constraint_penalty

        # Log the total reward for history tracking
        self.reward_history.append(total_reward)

        # Check if the episode has ended (e.g., reached the end of the time horizon)
        done = self.current_timestep >= self.H

        # Update the current state and timestep counter
        self.state = next_state
        self.current_timestep += 1

        return next_state, total_reward, done, {}

    def _calculate_reward(self, action, state):
        # Implement reward calculation logic or call an external function
        pass

    def _calculate_constraints(self, action, state):
        # Implement constraint checks and calculate penalties or call external functions
        pass

    def _update_state(self, action):
        # Implement state transition logic
        return new_state    

    def _check_done_condition(self):
        # Implement logic to check if the episode should end
        return False
    
    def _calculate_current_values(self, action):
        """
        Calculate P_Grid, P_gen, P_solar, P_wind, and curtailments based on action and state.
        """
        pass  # Implement power generation and curtailment logic
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