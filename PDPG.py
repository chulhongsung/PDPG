import numpy as np
import tensorflow as tf
from tensorflow import keras as K
import matplotlib.pyplot as plt

from collections import deque
import random

import gym

import datetime

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.count = 0

    def add_buffer(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)


        if self.count < self.buffer_size:
            self.buffer.append(transition)
            self.count += 1
        else: 
            self.buffer.popleft()
            self.buffer.append(transition)


    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        states = np.asarray([i[0] for i in batch])
        actions = np.asarray([i[1] for i in batch])
        rewards = np.asarray([i[2] for i in batch])
        next_states = np.asarray([i[3] for i in batch])
        dones = np.asarray([i[4] for i in batch])
        return states, actions, rewards, next_states, dones

    def buffer_count(self):
        return self.count

    def clear_buffer(self):
        self.buffer = deque()
        self.count = 0
        

class Critic(K.models.Model):
    """
    Critic network which returns parameters of the linear isotonic regression spline 

    Args:
        hidden_dims (list): A list consists of hidden layers output dimensions 
        param_dims (int): An integer value denotes the number of spline parameters such as knots(delta), slope(beta), ...
        min_reward (int): min_reward <= reward <= max_reward, if min_reward is not None, then parameter gamma = min_reward   
        inputs (tensor): tf.concat([temp_state, temp_action], axis=-1) 
    """
    
    def __init__(self, hidden_dims, param_dims, min_reward=None):
        super(Critic, self).__init__()   
        self.hidden_dims = hidden_dims
        self.min_reward = min_reward
        self.hidden_layers = [K.layers.Dense(hidden_dims[i]) for i in range(len(self.hidden_dims))]
        self.delta_map = K.layers.Dense(param_dims, activation='softmax')
        self.beta_map = K.layers.Dense(param_dims, activation='softplus')
    
        if self.min_reward is not None:
            self.gamma_map = tf.convert_to_tensor(self.min_reward, dtype=tf.float32)
        else:
            self.gamma_map = K.layers.Dense(1, activation='relu')
            
    def call(self, inputs):
        
        temp_h = inputs
        
        for i in range(len(self.hidden_dims)):
            temp_h = self.hidden_layers[i](temp_h)

        delta = self.delta_map(temp_h)
        beta = self.beta_map(temp_h)

        if self.min_reward is None:
            gamma = self.gamma_map(temp_h)
        else:
            gamma = self.gamma_map
        
        return delta, beta, gamma

class Actor(K.models.Model):

    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()

        self.action_bound = action_bound

        self.h1 = K.layers.Dense(400, activation='relu')
        self.h2 = K.layers.Dense(300, activation='relu')
        self.action = K.layers.Dense(action_dim, activation='tanh')


    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)

        a = self.action(x)
        
        a = K.layers.Lambda(lambda x: x*self.action_bound)(a)

        return a

class PDPGagent(object):
    
    def __init__(self, env):
        self.GAMMA = 0.95
        self.BATCH_SIZE = 50
        self.BUFFER_SIZE = 10000
        self.ACTOR_LEARNING_RATE = 0.001
        self.CRITIC_LEARNING_RATE = 0.001
        self.TAU = 0.001

        self.Q_levels = tf.constant(np.append([0.0], np.repeat(1, 20)/20), dtype=tf.dtypes.float32)

        self.env = env

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        self.actor = Actor(self.action_dim, self.action_bound)
        self.critic = Critic(hidden_dims=[200, 100], param_dims=21)

        self.actor.build(input_shape=(None, self.state_dim))

        self.optimizer_theta = K.optimizers.Adam(learning_rate=self.ACTOR_LEARNING_RATE )
        self.optimizer_phi = K.optimizers.Adam(learning_rate=self.CRITIC_LEARNING_RATE)

        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        self.save_epi_reward = []

    def ou_noise(self, x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        return x + rho*(mu - x)*dt + sigma*np.sqrt(dt)*np.random.normal(size=dim)

    @tf.function
    def linear_spline(self, alpha, gamma, beta, delta):
        """
        Linear isotonic regression spline function 
        
        Compute quantile z given quantile level alpha
        """

        mask = alpha - tf.math.cumsum(delta) >= 0
        
        bl = tf.concat([beta[0, tf.newaxis], (beta[1:] - beta[:-1])], axis=0)

        dl = tf.math.cumsum(delta)
        
        z = gamma + tf.math.reduce_sum(bl *  tf.cast(mask, dtype=tf.float32)) * alpha - tf.math.reduce_sum((bl * tf.cast(mask, dtype=tf.float32)) * (dl * tf.cast(mask, dtype=tf.float32)))
        
        return z

    def td_target(self, rewards, delta, beta, gamma, dones):
        y_k = np.zeros_like(delta)

        for i in range(delta.shape[0]): 
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = np.squeeze(rewards[i] + self.GAMMA * tf.map_fn(lambda x: self.linear_spline(x, gamma[i], beta[i], delta[i]), tf.math.cumsum(self.Q_levels)).numpy())
        return tf.clip_by_value(y_k, -1.0, 1.0)
    
    @tf.function
    def crps(self, z, gamma, beta, delta, quantile_dl):
        """
        Compute CRPS given samples z
        """

        mask = z >= quantile_dl
            
        bl = tf.concat([beta[0, tf.newaxis], (beta[1:] - beta[:-1])], axis=0)

        dl = tf.math.cumsum(delta)

        mask_bl = tf.boolean_mask(bl, mask)

        mask_dl = tf.boolean_mask(dl, mask)
        
        tilde_a =  tf.cast(tf.clip_by_value((z - gamma + tf.math.reduce_sum(mask_bl * mask_dl)) / (tf.math.reduce_sum(mask_bl)+0.000001), clip_value_min=0.001, clip_value_max=1), dtype=tf.dtypes.float32)
        
        crps_ = (2*tilde_a -1)*z + (1-2*tilde_a)*gamma + tf.math.reduce_sum(bl*((1/3)*(1 - dl**3) - dl - tf.maximum(tilde_a, dl)**2 + 2*tf.math.maximum(tilde_a, dl)*dl))
    
        return crps_

    def actor_critic_learn(self, states, rewards, next_states, dones):
       
        with tf.GradientTape(persistent=True) as tape:
          
            actions = self.actor(states)
            delta, beta, gamma = self.critic(tf.concat([states, actions], axis=-1))
            target_delta, target_beta, target_gamma = self.critic(tf.concat([tf.convert_to_tensor(next_states, dtype=tf.float32),
                                                self.actor(tf.convert_to_tensor(next_states, dtype=tf.float32))], axis=-1))
            td_targets = self.td_target(rewards, target_delta, target_beta, target_gamma, dones)
            
            temp_crps = 0
            
            z = 0
            
            M = delta.shape[0]
            
            for m in tf.range(M):
                w = tf.math.log(tf.math.cumsum(delta[m])+0.000001)/tf.math.reduce_sum(tf.math.log(tf.math.cumsum(delta[m])+0.000001))
                temp_quantile_dl = tf.clip_by_value(tf.squeeze(tf.map_fn(lambda x: self.linear_spline(x, gamma[m], beta[m], delta[m]), tf.math.cumsum(delta[m]))), -1.0, 1.0)
                temp_crps += tf.math.reduce_mean(tf.map_fn(lambda x: self.crps(x, gamma[m], beta[m], delta[m], temp_quantile_dl), td_targets[m])) / M
                z += tf.math.reduce_sum(-w * temp_quantile_dl) / M

        grad_theta = tape.gradient(z, self.actor.weights)
        
        grad_phi = tape.gradient(temp_crps, self.critic.weights)

        self.optimizer_theta.apply_gradients(zip(grad_theta, self.actor.weights))
        self.optimizer_phi.apply_gradients(zip(grad_phi, self.critic.weights))
    
    def train(self, MAX_EPISODE_NUM):

        for ep in range(MAX_EPISODE_NUM):

            pre_noise = np.zeros(self.action_dim)

            time, episode_reward, done = 0, 0, False

            state = self.env.reset()

            while not done:

                action = self.actor(tf.convert_to_tensor([state], dtype=tf.float32))
                action = action.numpy()[0]
                noise = self.ou_noise(pre_noise, dim=self.action_dim)

                action = np.clip(action + noise, -self.action_bound, self.action_bound)

                next_state, reward, done, _ = self.env.step(action)

                train_reward = (reward + 8) / 8

                self.buffer.add_buffer(state, action, train_reward, next_state, done)

                if self.buffer.buffer_count() > 1000:

                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.BATCH_SIZE)
                    states = tf.cast(states, dtype=tf.float32)
                    next_states = tf.cast(next_states, dtype=tf.float32)
                    self.actor_critic_learn(states, rewards, next_states, dones)

                pre_noise = noise
                state = next_state
                episode_reward += reward
                time += 1
            
            self.save_epi_reward.append(episode_reward)

            if (ep+1 % 50) == 0:
                self.actor.save_weights("./save_weights/pendulum_actor_" + datetime.date.today().strftime("%Y%m%d") + "_epoch" + str(ep+1) + ".h5")
                self.critic.save_weights("./save_weights/pendulum_critic_" + datetime.date.today().strftime("%Y%m%d") + "_epoch" + str(ep+1) + ".h5")
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)

    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()
        
try: import gym
except:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gym"])
    import gym

try: import pygame
except:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])
    import pygame

env = gym.make("Pendulum-v1")

agent = PDPGagent(env)

MAX_EPISODE_NUM = 300

agent.train(MAX_EPISODE_NUM)

agent.plot_result()

plt.savefig('/home1/prof/jeon/hong/pdpg/pendulum_v1_reward_batch' + str(agent.BATCH_SIZE) + '_epi' + str(MAX_EPISODE_NUM)  + '.png', dpi=300)
