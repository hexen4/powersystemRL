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
from microgrid_env import MicrogridEnv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from controllers.models import ActorPiModel, CriticQModel
from controllers.buffer import ExtendedPrioritizedReplayBuffer
from setting import *
import utils
import matplotlib.pyplot as plt
from IPython.display import clear_output

class TCSAC():
    def __init__(self, epsilon_p=0.001, **kwargs):
        
        self.counter = 0
        self.critic_weight = WEIGHT_CRITIC
        self.batch_size = BATCH_SIZE
        self.alpha = TEMP
        self.epsilon_p = epsilon_p

        # normalization
        self.obs_norm = utils.NormalizeObservation(STATE_SHAPE)

        # action bounds
        self.max_action = MAX_ACTION
        self.min_action = MIN_ACTION   
        self.action = None 

        # internal states
        self.prev_state = None
        self.action = None
        self.device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")

        
        # buffer
        self.buffer = ExtendedPrioritizedReplayBuffer()
        
        self.critic1 = CriticQModel(N_OBS).to(self.device)
        self.critic2 = CriticQModel(N_OBS).to(self.device)
        self.critic3 = CriticQModel(N_OBS).to(self.device)
        self.target_critic1 = CriticQModel(N_OBS).to(self.device)
        self.target_critic2 = CriticQModel(N_OBS).to(self.device)
        self.target_critic3 = CriticQModel(N_OBS).to(self.device)
        self.policy_net = ActorPiModel(N_OBS).to(self.device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=self.device)
        print('Soft Q Network (1,2,3): ', self.critic1)
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
        self.soft_q_optimizer3 = optim.Adam(self.critic3.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        
    
    def model_info(self):
        self.actor.summary()
        self.critic1.summary()
    
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

    def update(self, batch_size, reward_scale=1., auto_entropy=True, target_entropy=-2, gamma = DISCOUNT_FACTOR,_lambda=WEIGHT_CRITIC,soft_tau=TARGET_NETWORK_UPDATE):
        # sample
        state, action, reward, next_state,_ = self.buffer.sample(batch_size) #need to add weights
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        predicted_q_value3 = self.soft_q_net3(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state) 
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

    # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss  .backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
    # Compute target Q-values
        target_q1 = self.target_critic1(next_state, new_next_action)
        target_q23_min = torch.min(
            self.target_critic2(next_state, new_next_action),
            self.target_critic3(next_state, new_next_action)
        )
        target_q_min = _lambda * target_q1 + (1 - _lambda) * target_q23_min
        target_q_value = reward + (1 - done) * gamma * (target_q_min - self.alpha * next_log_prob)

        # Compute Q-function losses
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())
        q_value_loss3 = self.soft_q_criterion3(predicted_q_value3, target_q_value.detach())

    # Optimize Q-functions
        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        self.soft_q_optimizer3.zero_grad()
        q_value_loss3.backward()
        self.soft_q_optimizer3.step()

        # Generate new actions for policy update
        new_action, log_prob, _, _, _ = self.policy_net.evaluate(state)

        # Compute policy loss
        predicted_new_q_value = torch.min(
            self.soft_q_net1(state, new_action),
            self.soft_q_net2(state, new_action)
        )
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean() #if doesnt work then change signs here

        # Optimize policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        for target_param, param in zip(self.target_soft_q_net3.parameters(), self.soft_q_net3.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

        return predicted_new_q_value.mean()

    def plot(rewards):
        clear_output(True)
        plt.figure(figsize=(20,5))
        plt.plot(rewards)
        plt.savefig('sac_v2.png')
        plt.show()
    
    def train(self, max_episodes = MAX_EPISODES,max_steps = MAX_STEPS,batch_size = BATCH_SIZE):
        env = MicrogridEnv(w1=W1, w2=W2)
        rewards = []
        for eps in range(max_episodes):
            state = env.reset() 
            episode_reward = 0
            transitions = [None] * int(max_steps)
            for time in range(int(max_steps)):
                # Sample action from policy
                if self.counter > 0:
                    action = self.policy_net.get_action(state, deterministic=False)
                else:
                    action = self.policy_net.sample_action()
                # Step environment
                next_state, reward, done, _ = env.step(action,time)
                #next_state = self.obs_norm.normalize(state, update=False) # TODO do i need to normalise state?
                
                # Update networks if enough data is collected
                if self.buffer.__len__() > batch_size and self.counter > WARMUP and self.counter % UPDATE_FREQ == 0: 
                    self.update(batch_size)
                
                state = next_state
                episode_reward += reward
                transitions[time] = [torch.tensor(state), torch.tensor(action), torch.tensor(next_state), torch.tensor(reward)]
                self.counter += 1
                if done:
                    self.buffer.add_episodes(transitions,eps) 
                    break
                
            if eps % 20 == 0 and eps>0: # plot and model saving interval
                self.plot(rewards)
                np.save('rewards', rewards)
                TCSAC.save_model(model_path)
            print('Episode: ', eps, '| Episode Reward: ', episode_reward)
            rewards.append(episode_reward)
        self.save_model(model_path)

if __name__ == '__main__':
    tscac = TCSAC()
    tscac.train()