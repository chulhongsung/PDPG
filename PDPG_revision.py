import numpy as np
import tensorflow as tf
from tensorflow import keras as K
import matplotlib.pyplot as plt

from collections import deque
import random

import gym

import datetime

class ReplayBuffer(object):
    """
    Reply Buffer
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.count = 0

    def add_buffer(self, state, action, noise, reward, next_state, done):
        transition = (state, action, noise, reward, next_state, done)

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
        noises = np.asarray([i[2] for i in batch])
        rewards = np.asarray([i[3] for i in batch])
        next_states = np.asarray([i[4] for i in batch])
        dones = np.asarray([i[5] for i in batch])
        return states, actions, noises, rewards, next_states, dones

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
        self.beta_map = K.layers.Dense(param_dims, activation='softplus')
        self.gamma_map = K.layers.Dense(1, activation='tanh')                                
    
    def build(self, input_shape):
        self.batch_size = input_shape[0]
        self.delta = tf.repeat(tf.constant(np.append([0.0], np.repeat(1, 10)/10), dtype=tf.float32)[tf.newaxis, ...], self.batch_size, axis=0) 
        
    def call(self, inputs):
        
        temp_h = inputs
        
        for i in range(len(self.hidden_dims)):
            temp_h = self.hidden_layers[i](temp_h)

        beta = self.beta_map(temp_h)

        gamma = self.gamma_map(temp_h)
        
        w = tf.math.log(tf.math.cumsum(self.delta, axis=-1)+0.000001)/tf.math.reduce_sum(tf.math.log(tf.math.cumsum(self.delta, axis=-1)+0.000001), axis=-1)[..., tf.newaxis]
        
        return self.delta, beta, gamma, w

class Actor(K.models.Model):

    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()

        self.action_bound = action_bound

        self.h1 = K.layers.Dense(128, activation='relu')
        self.h2 = K.layers.Dense(64, activation='relu')
        self.h3 = K.layers.Dense(32, activation='relu')
        self.action = K.layers.Dense(action_dim, activation='tanh')


    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        a = self.action(x)

        a = K.layers.Lambda(lambda x: x*self.action_bound)(a)

        return a

class PDPGagent(object):
    
    def __init__(self, env):
        self.GAMMA_INIT = 0.3
        self.GAMMA = 0.3
        self.BATCH_SIZE = 256
        self.BUFFER_SIZE = 10000
        self.ACTOR_LEARNING_RATE = 0.00001
        self.CRITIC_LEARNING_RATE = 0.00002
        self.DELTA = 0.3
        self.DELTA_DECAY = 0.9999
        
        self.Q_levels = tf.constant(np.append([0.0], np.repeat(1, 10)/10), dtype=tf.float32)

        self.env = env

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        self.actor = Actor(self.action_dim, self.action_bound)
        self.critic = Critic(hidden_dims=[128, 64, 32], param_dims=11)

        self.actor.build(input_shape=(None, self.state_dim))
        
        self.optimizer_theta = K.optimizers.Adam(learning_rate=self.ACTOR_LEARNING_RATE)
        self.optimizer_phi = K.optimizers.Adam(learning_rate=self.CRITIC_LEARNING_RATE)

        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        self.save_epi_reward = []

    def gaussian_noise(self, mu=0.0, sigma=1):
        return tf.random.normal([1], mu, sigma, tf.float32)

    def linear_spline(self, arg):
        """
        Linear isotonic regression spline function 
        
        Compute quantile z given quantile level alpha
        """
        alpha, gamma, beta, delta = arg
        
        # dl = tf.concat((tf.zeros(shape=[1]), tf.math.cumsum(delta)[:-1]), axis=-1)
        
        dl = tf.math.cumsum(delta)
        
        mask = tf.nn.relu(tf.expand_dims(alpha, axis=-1) - tf.transpose(tf.expand_dims(dl, axis=-1)))
        
        bl = tf.transpose(tf.expand_dims(tf.concat([beta[0, tf.newaxis], (beta[1:] - beta[:-1])], axis=0), axis=-1))

        z = gamma + tf.math.reduce_sum(bl * mask, axis=-1)
        
        return z
    
    def knot_value(self, arg):
        """
        Linear isotonic regression spline function 
        
        Compute Knot-points values
        """
        gamma, beta, delta = arg
        
        bl = tf.concat([tf.expand_dims(beta[...,0], axis=-1), (beta[..., 1:] - beta[..., :-1])], axis=-1)
        
        dl = tf.math.cumsum(delta)
        
        a = tf.math.cumsum(bl * dl, axis=-1)

        b = tf.math.cumsum(bl, axis=-1) * dl

        z = b - a + gamma
        
        return z
    
    def crps(self, arg):
        """
        Compute CRPS given samples z
        """
        z, gamma, beta, delta, quantile_dl = arg
        
        mask =  tf.cast(tf.expand_dims(z, axis=-1) - tf.expand_dims(quantile_dl, axis=-2) > 0.0, tf.float32)
            
        bl = tf.concat([beta[0, tf.newaxis], (beta[1:] - beta[:-1])], axis=0)

        dl = tf.math.cumsum(delta)
        
        tilde_a = tf.clip_by_value((z - gamma + tf.reduce_sum(bl * dl * mask, axis=-1))/ (tf.reduce_sum(bl * mask, axis=-1) +0.000001), clip_value_min=0.0001, clip_value_max=1)
    
        crps_ = (2*tilde_a - 1)*z + (1-2*tilde_a)*gamma  + tf.math.reduce_sum(bl*((1/3)*(1 - dl**3) - dl - tf.maximum(tf.expand_dims(tilde_a, axis=-1), dl)**2 + 2*tf.math.maximum(tf.expand_dims(tilde_a, axis=-1), dl)*dl), axis=-1)
        
        return tf.reduce_mean(crps_)

    @tf.function
    def actor_critic_learn(self, states, noises, rewards, next_states, ep):
       
        with tf.GradientTape(persistent=True) as tape:
            
            actions = self.actor(states) + (self.DELTA_DECAY)**ep * self.DELTA * noises

            delta, beta, gamma, w = self.critic(tf.concat([states, actions], axis=-1))
            next_action = tf.clip_by_value(self.actor(tf.convert_to_tensor(next_states, dtype=tf.float32)) + (self.DELTA_DECAY)**ep * self.DELTA * self.gaussian_noise(), -self.action_bound , self.action_bound)
            target_delta, target_beta, target_gamma, _ = self.critic(tf.concat([tf.convert_to_tensor(next_states, dtype=tf.float32), next_action], axis=-1))
            td_targets =  tf.clip_by_value(tf.expand_dims(tf.cast(rewards, tf.float32), -1) + self.GAMMA * tf.vectorized_map(self.linear_spline, (tf.math.cumsum(tf.repeat(tf.expand_dims(self.Q_levels, -2), self.BATCH_SIZE, axis=0), axis=-1), target_gamma, target_beta, target_delta)), clip_value_min=-1.0, clip_value_max=1.0)  
            temp_quantile_dl = tf.clip_by_value(tf.vectorized_map(self.knot_value, (gamma, beta, delta)), -1.0, 1.0)
 
            crps = tf.math.reduce_mean(tf.vectorized_map(self.crps, (td_targets, gamma, beta, delta, temp_quantile_dl)))
            emp_upr = tf.math.reduce_mean(tf.expand_dims(-w, axis=-2) @ tf.expand_dims(temp_quantile_dl, axis=-1))

        grad_theta = tape.gradient(emp_upr, self.actor.weights)
        
        grad_phi = tape.gradient(crps, self.critic.weights)

        self.optimizer_theta.apply_gradients(zip(grad_theta, self.actor.weights))
        self.optimizer_phi.apply_gradients(zip(grad_phi, self.critic.weights))
        
        return grad_theta[0]
        
    def train(self, MAX_EPISODE_NUM):

        for ep in range(MAX_EPISODE_NUM):


            time, episode_reward, done = 0, 0, False

            state = self.env.reset()
            
            while not done:
                
                action = self.actor(tf.convert_to_tensor([state], dtype=tf.float32))
                action = action.numpy()[0]
                noise = self.gaussian_noise()

                action = np.clip(action + (self.DELTA_DECAY)**ep * self.DELTA * noise, -self.action_bound, self.action_bound)

                next_state, reward, done, _ = self.env.step(action)

                train_reward = (reward + 8) / 8

                self.buffer.add_buffer(state, action, noise, train_reward, next_state, done)

                if self.buffer.buffer_count() > 1000:
                    
                    states, actions, noises, rewards, next_states, dones = self.buffer.sample_batch(self.BATCH_SIZE)
                    states = tf.cast(states, dtype=tf.float32)
                    next_states = tf.cast(next_states, dtype=tf.float32)
                    grad_theta = self.actor_critic_learn(states, noises, rewards, next_states, ep)

                state = next_state
                episode_reward += reward
                time += 1
                
            self.save_epi_reward.append(round(episode_reward, 2))
            
            self.GAMMA = self.GAMMA_INIT * (1.000012)**(ep/10)                
            
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', round(episode_reward, 2))
            
            
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
#%%
env = gym.make("Pendulum-v1")

agent = PDPGagent(env)

MAX_EPISODE_NUM = 1000000

agent.train(MAX_EPISODE_NUM)

agent.plot_result()
