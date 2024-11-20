# generate action -> q-networks -> target q value -> q-network updates -> policy network soft update
'''
class:
- TD3Agent
    - adapt_param_noise()
    - adjust_action_noise()
    - calculate_distance()
    - calculate_reward()
    - control_step()
    - get_state()
    - is_converged()
    - learn()
    - load_models()
    - model_info()
    - perturb_policy()
    - policy()
    - reset()
    - save_models()
    - update_actor()
    - update_critics()
    - update_sequence_model()
    - update_target_networks()
    - time_step()
'''
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
import utils

class TD3Agent(Controller):
    def __init__(self, net, ids, pv_profile_df, wt_profile_df, load_profile_df, price_profile_df,
        noise_type = 'action', sequence_model_type='none', use_pretrained_sequence_model=False,
        n_epochs=None, training=False,
        delay=2, gamma=GAMMA, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, 
        buffer_size=50000, batch_size=128, epsilon_p=0.001, **kwargs):
        super().__init__(net, **kwargs)
        self.ids = ids
        self.pv_profile = pv_profile_df
        self.wt_profile = wt_profile_df
        self.load_profile = load_profile_df
        self.price_profile = price_profile_df
        self.state_seq_shape = STATE_SEQ_SHAPE
        self.state_fnn_shape = STATE_FNN_SHAPE
        self.n_action = N_ACTION
        self.use_pretrained_sequence_model = use_pretrained_sequence_model
        self.training = training
        self.noise_type = noise_type
        self.action_noise_scale = ACTION_NOISE_SCALE
        self.action_noise_scale_ = ACTION_NOISE_SCALE
        self.param_noise_adapt_rate = PARAM_NOISE_ADAPT_RATE
        self.param_noise_bound = PARAM_NOISE_BOUND
        self.param_noise_scale = PARAM_NOISE_SCALE
        self.n_epochs = n_epochs
        self.update_freq = UPDATE_FREQ
        self.update_times = UPDATE_TIMES
        self.warmup = WARMUP
        self.delay = delay
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_p = epsilon_p

        # counter
        self.time_step_counter = 0
        self.learn_step_counter = 0

        # normalization
        self.obs_norm = utils.NormalizeObservation()
        self.a_norm = utils.NormalizeAction()
        self.r_norm = utils.NormalizeReward()

        # TODO add battery IDS here when implemented
        self.pv_id = ids.get('pv')
        self.pv_p_mw = net.sgen.at[self.pv_id, 'p_mw']
        self.wt_id = ids.get('wt')
        self.wt_p_mw = net.gen.at[self.wt_id, 'p_mw']
        self.cdg_id = ids.get('dg')
        self.cdg_p_mw = net.gen.at[self.dg_id, 'p_mw']
        self.c1_id  = id.get('c1')
        self.c2_id  = id.get('c2')
        self.c3_id  = id.get('c3')
        self.c4_id  = id.get('c4')
        self.c5_id  = id.get('c5')

        # action bounds
        self.max_action = MAX_ACTION
        self.min_action = MIN_ACTION    

        # internal states
        self.prev_state = None
        self.bat5_soc = 0.
        self.bat10_soc = 0.
        self.action = None
        self.rewards = []
        self.costs = []
        self.history = {
            'price': [],
            'excess': [],
        }
        self.last_time_step = None
        self.applied = False

        
        # buffer
        # self.buffer = ReplayBuffer(buffer_size, self.state_seq_shape, self.state_fnn_shape, self.n_action)
        self.buffer = PrioritizedReplayBuffer(buffer_size, self.state_seq_shape, self.state_fnn_shape, self.n_action)

        # models
        self.sequence_model_type = sequence_model_type
        if self.sequence_model_type == 'none': 
            # actor critic
            self.actor = get_mu_actor(SequenceModel(sequence_model_type, name='sequence_model'), ActorMuModel(self.n_action))
            self.perturbed_actor = get_mu_actor(SequenceModel(sequence_model_type, name='sequence_model'), ActorMuModel(self.n_action))
            self.target_actor = get_mu_actor(SequenceModel(sequence_model_type, name='sequence_model'), ActorMuModel(self.n_action))
            self.critic1 = get_q_critic(SequenceModel(sequence_model_type, name='sequence_model'), CriticQModel())
            self.critic2 = get_q_critic(SequenceModel(sequence_model_type, name='sequence_model'), CriticQModel())
            self.target_critic1 = get_q_critic(SequenceModel(sequence_model_type, name='sequence_model'), CriticQModel())
            self.target_critic2 = get_q_critic(SequenceModel(sequence_model_type, name='sequence_model'), CriticQModel())
            # TODO add third critic
            # TODO add 3 actors??
        else: #evals to this
            # sequence model
            self.sequence_model = SequenceModel(sequence_model_type, name='sequence_model')
            self.sequence_model.compile(optimizer=Adam(learning_rate=lr_critic, epsilon=1e-5))

            # actor critic
            self.actor = get_mu_actor(self.sequence_model, ActorMuModel(self.n_action))
            self.perturbed_actor = get_mu_actor(self.sequence_model, ActorMuModel(self.n_action))
            self.target_actor = get_mu_actor(self.sequence_model, ActorMuModel(self.n_action))
            self.critic1 = get_q_critic(self.sequence_model, CriticQModel())
            self.critic2 = get_q_critic(self.sequence_model, CriticQModel())
            self.target_critic1 = get_q_critic(self.sequence_model, CriticQModel())
            self.target_critic2 = get_q_critic(self.sequence_model, CriticQModel())

        self.actor.compile(optimizer=Adam(learning_rate=lr_actor, epsilon=1e-5))
        self.critic1.compile(optimizer=Adam(learning_rate=lr_critic, epsilon=1e-5))
        self.critic2.compile(optimizer=Adam(learning_rate=lr_critic, epsilon=1e-5))