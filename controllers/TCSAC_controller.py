# generate action -> q-networks -> target q value -> q-network updates -> policy network soft update
'''
class:
- TCSAC
    - adjust_action_noise() done for me?
    - calculate_distance() done for me
    - calculate_reward() done
    - control_step() done
    - get_state() done
    - is_converged() done
    - learn() done
    - load_models() done for me 
    - model_info() done for me
    - perturb_policy() not done (entropy)
    - policy() done (samples mean and std from heavy sampling function, and reparam trick)
    - reset() done
    - save_models() done for me
    - update() not done 
    - update_actor() not done 
    - update_critics() not done
    - update_sequence_model() not done 
    - update_target_networks() not done
    - time_step() done
'''
import logging
import os
from typing import Dict
import math
import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from pandapower.control.basic_controller import Controller
from controllers.models import ActorPiModel, CriticQModel, CriticVModel
from controllers.buffer import ComprehensivePrioritizedReplayBuffer
from setting import *
import utils
from IPython.display import clear_output
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()

class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action
    

class TCSAC(Controller):
    def __init__(self, net, ids, epsilon_p=0.001, **kwargs):
        super().__init__(net, **kwargs)
        self.ids = ids
        self.curtailment_indices = [6, 19, 11, 27, 22]

        self.update_freq = UPDATE_FREQ
        self.update_times = UPDATE_TIMES
        self.critic_weight = WEIGHT_CRITIC
        self.batch_size = BATCH_SIZE
        self.delay = 2
        self.buffer_size = BUFFER_SIZE
        self.epsilon_p = epsilon_p

        # normalization
        self.obs_norm = utils.NormalizeObservation()
        self.a_norm = utils.NormalizeAction()
        self.r_norm = utils.NormalizeReward()

        # action bounds
        self.max_action = MAX_ACTION
        self.min_action = MIN_ACTION   
        self.action = None 

        # internal states
        self.prev_state = None
        self.action = None
        

        
        # buffer
        self.buffer = ComprehensivePrioritizedReplayBuffer()
        
        self.critic1 = CriticQModel(N_OBS).to(device)
        self.critic2 = CriticQModel(N_OBS).to(device)
        self.critic3 = CriticQModel(N_OBS).to(device)
        self.target_critic1 = CriticQModel(N_OBS).to(device)
        self.target_critic2 = CriticQModel(N_OBS).to(device)
        self.target_critic3 = CriticQModel(N_OBS).to(device)
        self.policy_net = ActorPiModel(N_OBS).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print('Soft Q Network (1,2,3): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic3.parameters(), self.critic3.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        self.soft_q_criterion3 = nn.MSELoss()
        
        soft_q_lr = 1e-3
        policy_lr = 1e-3
        alpha_lr  = 1e-3

        self.soft_q_optimizer1 = optim.Adam(self.critic1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.critic2.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer3 = optim.Adam(self.critic.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        
    
    def model_info(self):
        self.actor.summary()
        self.critic1.summary()

    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample(mean.shape).to(device)
        action = self.action_range* torch.tanh(mean + std*z)
        
        action = self.action_range* torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else action.detach().cpu().numpy()[0]
        return action

    def sample_action(self,):
        a=torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return self.action_range*a.numpy()

    def policy(self, state, random, deterministic=False):
        # network outputs
        state = self.obs_norm.normalize(state, update=False) 
        if not random:
            # reparam trick
            nn_action = self.get_action(state, deterministic)
        else: 
        # Deterministic action during testing (mean of the distribution)
            nn_action = self.sample_action
        
        nn_action = np.clip(nn_action, -NN_BOUND, NN_BOUND)

        # Extract curtailment actions and incentive rate from nn_action
        curtailment_actions = nn_action[:-1]  
        incentive_rate_action = nn_action[-1]  


        scaled_curtailments = utils.scale_action(curtailment_actions, MIN_ACTION[:-1], MAX_ACTION[:-1])
        scaled_incentives = utils.scale_action(incentive_rate_action, MIN_ACTION[-1], MAX_ACTION[-1])
        # Combine scaled curtailments and incentive rate into mg_action
        mg_action = np.concatenate((scaled_curtailments, [scaled_incentives]))

        return mg_action, nn_action
    
    def save_models(self, path):
        print('... Saving Models ...')
        torch.save(self.policy_net.state_dict(), f'{path}_actor_weights')
        torch.save(self.critic1.state_dict(), f'{path}_critic1_weights')
        torch.save(self.critic2.state_dict(), f'{path}_critic2_weights')
        torch.save(self.critic3.state_dict(), f'{path}_critic3_weights')
        torch.save(self.target_critic1.state_dict(), f'{path}_target_critic1_weights')
        torch.save(self.target_critic2.state_dict(), f'{path}_target_critic2_weights')
        torch.save(self.target_critic3.state_dict(), f'{path}_target_critic3_weights')

    def load_models(self, path):
        print('... Loading Models ...')
        self.policy_net.load_state_dict(torch.load(f'{path}_actor_weights'))
        self.critic1.load_state_dict(torch.load(f'{path}_critic1_weights'))
        self.critic2.load_state_dict(torch.load(f'{path}_critic2_weights'))
        self.critic3.load_state_dict(torch.load(f'{path}_critic3_weights'))
        self.target_critic1.load_state_dict(torch.load(f'{path}_target_critic1_weights'))
        self.target_critic2.load_state_dict(torch.load(f'{path}_target_critic2_weights'))
        self.target_critic3.load_state_dict(torch.load(f'{path}_target_critic3_weights'))

        # Set models to evaluation mode
        self.policy_net.eval()
        self.critic1.eval()
        self.critic2.eval()
        self.critic3.eval()
        self.target_critic1.eval()
        self.target_critic2.eval()
        self.target_critic3.eval()

# if len(replay_buffer) > batch_size and timestep > warmup and timestep % update_freq == 0:
#     for i in range(update_itr):
#         sac_trainer.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY, target_entropy=-1.*action_dim)
    def update(self): 
        # sample
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size) #need to add weights
        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        predicted_q_value3 = self.soft_q_net3(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state) # TODO start here -> need to change into diffferent
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

        # update critics
        critic_loss, target_values, td_errs = self.update_critics(state_seq_batch, state_fnn_batch, action_batch, reward_batch, next_state_seq_batch, next_state_fnn_batch, weights)
        self.update_priority(td_errs, idxs)
        self.learn_step_counter += 1

        # update sequence model
        if (self.sequence_model_type != 'none') and (not self.use_pretrained_sequence_model):
            self.update_sequence_model(state_seq_batch, state_fnn_batch, action_batch, target_values)

        if self.learn_step_counter % self.delay != 0:
            return

        # update actor
        actor_loss = self.update_actor(state_seq_batch, state_fnn_batch)

        # parameter noise
        if self.noise_type == 'param':
            self.perturb_policy()
            d = self.calculate_distance(state_seq_batch, state_fnn_batch)
            self.adapt_param_noise(d)

        # update targets
        self.update_target_networks()

    @tf.function
    def update_actor(self, state_seq_batch, state_fnn_batch):
        # trainable variables
        if self.sequence_model_type == 'none':
            actor_vars = self.actor.trainable_variables
        else:
            actor_vars = self.actor.get_layer('actor_mu_model').trainable_variables

        # gradient descent
        with tf.GradientTape() as tape:
            actions = self.actor([state_seq_batch, state_fnn_batch], training=True)
            actions = self.a_norm.tf_normalize(actions)
            q_values = self.critic1([state_seq_batch, state_fnn_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(q_values)
        actor_grads = tape.gradient(actor_loss, actor_vars)
        self.actor.optimizer.apply_gradients(zip(actor_grads, actor_vars))

        return actor_loss
    
    @tf.function
    def update_critics(self, state_seq_batch, state_fnn_batch, action_batch, reward_batch, next_state_seq_batch, next_state_fnn_batch, weights):
        # Issue: https://github.com/tensorflow/tensorflow/issues/35928
        # with tf.GradientTape(persistent=True) as tape:

        # target actions
        target_actions = self.target_actor([next_state_seq_batch, next_state_fnn_batch], training=True)
        #target_actions += tf.clip_by_value(tf.random.normal(shape=(self.batch_size, self.n_action), stddev=0.2), -0.5, 0.5) this is noise
        target_actions = tf.clip_by_value(target_actions, -NN_BOUND, NN_BOUND)
        target_actions = self.a_norm.tf_normalize(target_actions)

        # target values
        target_q_value1 = self.target_critic1([next_state_seq_batch, next_state_fnn_batch, target_actions], training=True)
        target_q_value2 = self.target_critic2([next_state_seq_batch, next_state_fnn_batch, target_actions], training=True)
        target_q_value3 = self.target_critic2([next_state_seq_batch, next_state_fnn_batch, target_actions], training=True)
        target_values = reward_batch + (self.critic_weight *target_q_value1) + (1-self.critic_weight)*tf.math.minimum(target_q_value2, target_q_value3) #need entropy term

        # td errors
        td_errs = target_values - self.critic1([state_seq_batch, state_fnn_batch, action_batch])

        # trainable variables
        if self.sequence_model_type == 'none':
            critic1_vars = self.critic1.trainable_variables
            critic2_vars = self.critic2.trainable_variables
        else:
            critic1_vars = self.critic1.get_layer('critic_q_model').trainable_variables
            critic2_vars = self.critic2.get_layer('critic_q_model_1').trainable_variables

        huber_loss = keras.losses.Huber()
        # update critic model 1
        with tf.GradientTape() as tape1:
            critic_loss1 = huber_loss(weights*target_values, weights*self.critic1([state_seq_batch, state_fnn_batch, action_batch], training=True))
        critic_grads1 = tape1.gradient(critic_loss1, critic1_vars)
        self.critic1.optimizer.apply_gradients(zip(critic_grads1, critic1_vars))

        # update critic model 2
        with tf.GradientTape() as tape2:
            critic_loss2 = huber_loss(weights*target_values, weights*self.critic2([state_seq_batch, state_fnn_batch, action_batch], training=True))
        critic_grads2 = tape2.gradient(critic_loss2, critic2_vars)
        self.critic2.optimizer.apply_gradients(zip(critic_grads2, critic2_vars))

        return critic_loss1, target_values, td_errs
    
    def update_priority(self, td_errs, idxs):
        priorities = np.abs(td_errs.numpy().flatten()) + self.epsilon_p
        for idx, p in zip(idxs, priorities):
            self.buffer.update_tree(idx, p)


    