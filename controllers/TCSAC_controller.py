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
    - policy() not done 
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
        self.curtailment_indices = [6, 19, 11, 27, 22]

        self.state_seq_shape = STATE_SEQ_SHAPE
        self.state_fnn_shape = STATE_FNN_SHAPE
        self.n_action = N_CONTROLLABLE_STATES

        self.use_pretrained_sequence_model = use_pretrained_sequence_model
        self.training = training

        self.noise_type = noise_type
        self.action_noise_scale = ACTION_NOISE_SCALE

        self.n_epochs = n_epochs
        self.update_freq = UPDATE_FREQ
        self.update_times = UPDATE_TIMES
        self.warmup = WARMUP
        self.delay = delay
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_p = epsilon_p
        self.action = None

        # counter
        self.timestep = TIMESTEPS

        # normalization
        self.obs_norm = utils.NormalizeObservation()
        self.a_norm = utils.NormalizeAction()
        self.r_norm = utils.NormalizeReward()

        #i dont think ids are neccessary 
        # action bounds
        self.max_action = MAX_ACTION
        self.min_action = MIN_ACTION    

        # internal states
        self.prev_state = None
        self.action = None
        self.timestep = TIMESTEPS
        self.applied = False

        
        # buffer
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

    def control_step(self, net, line_losses):
        if not self.applied:
            load_values = net.load.loc[self.curtailment_indices, 'p_mw'].to_numpy()
            updated_load_values = np.maximum(load_values - np.array(self.action[:-1]), 0)
            net.load.loc[self.curtailment_indices, 'p_mw'] = updated_load_values
            net.gen.loc[self.ids.get('dg'), 'p_mw'] = line_losses
            self.applied = True
        else:
            print("misaligned control step")
    
    def learn(self):
        if self.buffer.buffer_counter < self.batch_size:
            return

        if self.buffer.buffer_counter < self.warmup:
            return

        if self.timestep % self.update_freq != 0:
            return

        for _ in range(self.update_times):
            self.update()
        
    def load_models(self, dir='model_weights', run=1):
        print('... Loading Models ...')
        self.actor.load_weights(os.path.join(dir, self.sequence_model_type, str(run), 'actor_weights'))
        self.critic1.load_weights(os.path.join(dir, self.sequence_model_type, str(run), 'critic1_weights'))
        self.critic2.load_weights(os.path.join(dir, self.sequence_model_type, str(run), 'critic2_weights'))
        self.target_actor.load_weights(os.path.join(dir, self.sequence_model_type, str(run), 'target_actor_weights'))
        self.target_critic1.load_weights(os.path.join(dir, self.sequence_model_type, str(run), 'target_critic1_weights'))
        self.target_critic2.load_weights(os.path.join(dir, self.sequence_model_type, str(run), 'target_critic2_weights'))

    def model_info(self):
        self.actor.summary()
        self.critic1.summary()
    
    @tf.function
    def perturb_policy(self): #this is exploration -> entropy!!!
        pass
    
    def policy(self, net, t, state):
        # network outputs
        if self.timestep < self.warmup and self.training:
            # warmup
            nn_action = np.random.uniform(low=-NN_BOUND, high=NN_BOUND, size=(self.n_action,))
        else:
            state_seq, state_fnn = self.obs_norm.normalize(state, update=False)
            # add batch index
            tf_state_seq = tf.expand_dims(tf.convert_to_tensor(state_seq, dtype=tf.float32), axis=0) 
            tf_state_fnn = tf.expand_dims(tf.convert_to_tensor(state_fnn, dtype=tf.float32), axis=0)

            if self.training:
                # param noise
                if self.noise_type == 'param':
                    tf_action = self.perturbed_actor([tf_state_seq, tf_state_fnn], training=self.training)
                    tf_action = tf.squeeze(tf_action, axis=0) # remove batch index
                    nn_action = tf_action.numpy()
                # action noise
                else:
                    tf_action = self.actor([tf_state_seq, tf_state_fnn], training=self.training)
                    tf_action = tf.squeeze(tf_action, axis=0) # remove batch index
                    nn_action = tf_action.numpy()
                    if t % 100 == 0:
                        print(f'nn outputs = {nn_action}')
                    nn_action += np.random.normal(loc=0., scale=self.action_noise_scale, size=(self.n_action,))
                # testing
            else:
                tf_action = self.actor([tf_state_seq, tf_state_fnn], training=self.training)
                tf_action = tf.squeeze(tf_action, axis=0) # remove batch index
                nn_action = tf_action.numpy()
            nn_action = np.clip(nn_action, -NN_BOUND, NN_BOUND)

        # mg action
        p_b5_min = max(P_B5_MIN, (SOC_MIN - self.bat5_soc) * self.bat5_max_e_mwh / HOUR_PER_TIME_STEP)
        p_b5_max = min(P_B5_MAX, (SOC_MAX - self.bat5_soc) * self.bat5_max_e_mwh / HOUR_PER_TIME_STEP)
        p_b10_min = max(P_B10_MIN, (SOC_MIN - self.bat10_soc) * self.bat10_max_e_mwh / HOUR_PER_TIME_STEP)
        p_b10_max = min(P_B10_MAX, (SOC_MAX - self.bat10_soc) * self.bat10_max_e_mwh / HOUR_PER_TIME_STEP)

        # invalid action clipping
        # mgt5_p_mw, mgt9_p_mw, mgt10_p_mw, bat5_p_mw, bat10_p_mw = utils.scale_to_mg(nn_action, self.max_action, self.min_action)
        # bat5_p_mw = np.clip(bat5_p_mw, p_b5_min, p_b5_max)
        # bat10_p_mw = np.clip(bat10_p_mw, p_b10_min, p_b10_max)

        # invalid action masking


        self.min_action[ACTION_IDX.get('p_b5')] = p_b5_min
        self.min_action[ACTION_IDX.get('p_b10')] = p_b10_min
        self.max_action[ACTION_IDX.get('p_b5')] = p_b5_max
        self.max_action[ACTION_IDX.get('p_b10')] = p_b10_max
        bat5_p_mw, bat10_p_mw = utils.scale_to_mg(nn_action, self.min_action, self.max_action)

        mg_action = np.array([bat5_p_mw, bat10_p_mw])
        self.timestep += 1
        return mg_action, nn_action
    
    def save_models(self, dir='model_weights', run=1):
        print('... Saving Models ...')
        self.actor.save_weights(os.path.join(dir, self.sequence_model_type, str(run), 'actor_weights'))
        self.critic1.save_weights(os.path.join(dir, self.sequence_model_type, str(run), 'critic1_weights'))
        self.critic2.save_weights(os.path.join(dir, self.sequence_model_type, str(run), 'critic2_weights'))
        self.target_actor.save_weights(os.path.join(dir, self.sequence_model_type, str(run), 'target_actor_weights'))
        self.target_critic1.save_weights(os.path.join(dir, self.sequence_model_type, str(run), 'target_critic1_weights'))
        self.target_critic2.save_weights(os.path.join(dir, self.sequence_model_type, str(run), 'target_critic2_weights'))

    def update(self):
        # sample
        state_seq_batch, state_fnn_batch, action_batch, reward_batch, next_state_seq_batch, next_state_fnn_batch, idxs, weights = self.buffer.sample(self.batch_size)
        state_seq_batch = tf.convert_to_tensor(state_seq_batch, dtype=tf.float32)
        state_fnn_batch = tf.convert_to_tensor(state_fnn_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        next_state_seq_batch = tf.convert_to_tensor(next_state_seq_batch, dtype=tf.float32)
        next_state_fnn_batch = tf.convert_to_tensor(next_state_fnn_batch, dtype=tf.float32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)

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
        target_actions += tf.clip_by_value(tf.random.normal(shape=(self.batch_size, self.n_action), stddev=0.2), -0.5, 0.5)
        target_actions = tf.clip_by_value(target_actions, -NN_BOUND, NN_BOUND)
        target_actions = self.a_norm.tf_normalize(target_actions)

        # target values
        target_q_value1 = self.target_critic1([next_state_seq_batch, next_state_fnn_batch, target_actions], training=True)
        target_q_value2 = self.target_critic2([next_state_seq_batch, next_state_fnn_batch, target_actions], training=True)
        target_values = reward_batch + self.gamma * tf.math.minimum(target_q_value1, target_q_value2)

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

    @tf.function
    def update_sequence_model(self, state_seq_batch, state_fnn_batch, action_batch, target_values):
        huber_loss = keras.losses.Huber()
        with tf.GradientTape() as tape:
            critic_loss = huber_loss(target_values, self.critic1([state_seq_batch, state_fnn_batch, action_batch], training=True)) 
            critic_loss += huber_loss(target_values, self.critic2([state_seq_batch, state_fnn_batch, action_batch], training=True))
            critic_loss /= (2 * SEQ_LENGTH)
        seq_grads = tape.gradient(critic_loss, self.sequence_model.trainable_variables)
        seq_grads = [tf.clip_by_norm(g, 1.0) for g in seq_grads]
        self.sequence_model.optimizer.apply_gradients(zip(seq_grads, self.sequence_model.trainable_variables))
    
    @tf.function
    def update_target_networks(self, tau=0.005):
        if self.sequence_model_type == 'none':
            target_actor_weights = self.target_actor.trainable_weights
            actor_weights = self.actor.trainable_weights
            target_critic1_weights = self.target_critic1.trainable_weights
            critic1_weights = self.critic1.trainable_weights
            target_critic2_weights = self.target_critic2.trainable_weights
            critic2_weights = self.critic2.trainable_weights
        else:
            target_actor_weights = self.target_actor.get_layer('actor_mu_model_2').trainable_weights
            actor_weights = self.actor.get_layer('actor_mu_model').trainable_weights
            target_critic1_weights = self.target_critic1.get_layer('critic_q_model_2').trainable_weights
            critic1_weights = self.critic1.get_layer('critic_q_model').trainable_weights
            target_critic2_weights = self.target_critic2.get_layer('critic_q_model_3').trainable_weights
            critic2_weights = self.critic2.get_layer('critic_q_model_1').trainable_weights

        # update target actor
        for target_weight, weight in zip(target_actor_weights, actor_weights):
            target_weight.assign(tau * weight + (1 - tau) * target_weight)

        # update target critic1
        for target_weight, weight in zip(target_critic1_weights, critic1_weights):
            target_weight.assign(tau * weight + (1 - tau) * target_weight)

        # update target critic2
        for target_weight, weight in zip(target_critic2_weights, critic2_weights):
            target_weight.assign(tau * weight + (1 - tau) * target_weight)