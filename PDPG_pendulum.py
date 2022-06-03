import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as K
import matplotlib.pyplot as plt

from collections import deque
import random

import gym

class ReplayBuffer(object):
    """
    Reply Buffer
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.count = 0

    ## 버퍼에 저장
    def add_buffer(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)

        # 버퍼가 꽉 찼는지 확인
        if self.count < self.buffer_size:
            self.buffer.append(transition)
            self.count += 1
        else: # 찼으면 가장 오래된 데이터 삭제하고 저장
            self.buffer.popleft()
            self.buffer.append(transition)

    ## 버퍼에서 데이터 무작위로 추출 (배치 샘플링)
    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        # 상태, 행동, 보상, 다음 상태별로 정리
        states = np.asarray([i[0] for i in batch])
        actions = np.asarray([i[1] for i in batch])
        rewards = np.asarray([i[2] for i in batch])
        next_states = np.asarray([i[3] for i in batch])
        dones = np.asarray([i[4] for i in batch])
        return states, actions, rewards, next_states, dones


    ## 버퍼 사이즈 계산
    def buffer_count(self):
        return self.count


    ## 버퍼 비움
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
        self.hidden_layers = [K.layers.Dense(hidden_dims[i], activation='relu', kernel_initializer=K.initializers.GlorotUniform()) for i in range(len(self.hidden_dims))]
        self.hidden_state  = K.layers.Dense(64, activation='relu', kernel_initializer=K.initializers.GlorotUniform()) 
        self.hidden_action = K.layers.Dense(64, activation='relu', kernel_initializer=K.initializers.GlorotUniform())
        self.beta_map = K.layers.Dense(param_dims, activation='softplus', kernel_initializer=K.initializers.GlorotUniform(), bias_initializer=
                        K.initializers.random_uniform(minval=-0.0001, maxval=0.0001))
        self.gamma_map = K.layers.Dense(1, activation='tanh', kernel_initializer=K.initializers.GlorotUniform())                                
    
    def build(self, input_shape):
        self.batch_size = input_shape[0][0]
        self.delta = tf.repeat(tf.constant(np.append([0.0], np.repeat(1, 10)/10), dtype=tf.float32)[tf.newaxis, ...], self.batch_size, axis=0) 
   
    @tf.function    
    def call(self, inputs):
        
        states, actions = inputs
        
        x = self.hidden_state(states)
        a = self.hidden_action(actions)        
                
        temp_h = K.layers.concatenate([x, a], axis=-1)
        
        for i in range(len(self.hidden_dims)):
            temp_h = self.hidden_layers[i](temp_h)

        # delta = self.delta_map(temp_h)
        beta = self.beta_map(temp_h)
        
        gamma = self.gamma_map(temp_h)
        
        return self.delta, beta, gamma

class Actor(K.models.Model):

    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()

        self.action_bound = action_bound

        self.h1 = K.layers.Dense(128, activation='relu',kernel_initializer=K.initializers.GlorotUniform())
        self.h2 = K.layers.Dense(64, activation='relu', kernel_initializer=K.initializers.GlorotUniform())
        #self.h3 = K.layers.Dense(32, activation='relu')
        self.action = K.layers.Dense(action_dim, activation='tanh', kernel_initializer=K.initializers.GlorotUniform())

    @tf.function 
    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        #x = self.h3(x)
        a = self.action(x)

        # 행동을 [-action_bound, action_bound] 범위로 조정
        a = K.layers.Lambda(lambda x: x*self.action_bound)(a)

        return a

class PDPGagent(object):
    
    def __init__(self, env):
        self.GAMMA = 0.9
        self.BATCH_SIZE = 256
        self.BUFFER_SIZE = 1000000
        self.ACTOR_LEARNING_RATE = 0.00003
        self.CRITIC_LEARNING_RATE = 0.00003
        self.DELTA = 0.3
        self.DELTA_DECAY = 0.9999
        self.ETA = 3
        self.ETA_DECAY = 0.9999
        self.QUANTILE_SAMPLES = 36
        self.TAU = 0.001
        
        self.q_levels = tf.constant(np.append([0.0], np.repeat(1, 10)/10), dtype=tf.float32)

        self.env = env

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        self.actor = Actor(self.action_dim, self.action_bound)
        self.target_actor = Actor(self.action_dim, self.action_bound)
        
        self.critic = Critic(hidden_dims=[300, 200], param_dims=11)
        self.target_critic = Critic(hidden_dims=[300, 200], param_dims=11)

        self.actor.build(input_shape=(None, self.state_dim))
        self.target_actor.build(input_shape=(None, self.state_dim))
        
        self.optimizer_theta = K.optimizers.Adam(learning_rate=self.ACTOR_LEARNING_RATE)
        self.optimizer_phi = K.optimizers.Adam(learning_rate=self.CRITIC_LEARNING_RATE)

        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        # 에피소드에서 얻은 총 보상값을 저장하기 위한 변수
        self.save_epi_reward = []

    def gaussian_noise(self, mu=0.0, sigma=1):
        return tf.random.normal([1], mu, sigma, tf.float32)
    

    def linear_spline(self, arg):
        """
        Linear isotonic regression spline function 
        
        Compute quantile z given quantile level alpha
        """
        alpha, gamma, beta, delta = arg
        
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

    def update_target_network(self):
        theta = self.actor.get_weights()
        target_theta = self.target_actor.get_weights()
        for i in range(len(theta)):
            target_theta[i] = self.TAU * theta[i] + (1 - self.TAU) * target_theta[i]
        self.target_actor.set_weights(target_theta)

        phi = self.critic.get_weights()
        target_phi = self.target_critic.get_weights()
        for i in range(len(phi)):
            target_phi[i] = self.TAU * phi[i] + (1 - self.TAU) * target_phi[i]
        self.target_critic.set_weights(target_phi)
        
    
    @tf.function
    def actor_learn(self, states):
        with tf.GradientTape() as tape:
            temp_delta, temp_beta, temp_gamma = self.critic([states, self.actor(states)])
            temp_w = tf.math.log(tf.math.cumsum(self.q_levels, axis=-1)+0.000001)/tf.math.reduce_sum(tf.math.log(tf.math.cumsum(self.q_levels, axis=-1)+0.000001), axis=-1)[..., tf.newaxis]
            temp_quantile_dl = tf.vectorized_map(self.knot_value, (temp_gamma, temp_beta, temp_delta))
            emp_upr = tf.math.reduce_mean(tf.math.reduce_sum(-temp_w * temp_quantile_dl, axis=-1)) 
    
        grad_theta = tape.gradient(emp_upr, self.actor.weights)
        self.optimizer_theta.apply_gradients(zip(grad_theta, self.actor.weights))
        
        return grad_theta[0]
        
    @tf.function 
    def critic_learn(self, states, actions, rewards, next_states):
        with tf.GradientTape() as tape:            
            delta, beta, gamma = self.critic([states, actions])
            next_action = self.target_actor(tf.convert_to_tensor(next_states, dtype=tf.float32))
            target_delta, target_beta, target_gamma = self.target_critic([tf.convert_to_tensor(next_states, dtype=tf.float32), next_action])
            temp_alpha = (tf.random.uniform([256, 24], minval=0.0, maxval=1.0) + tf.random.uniform([256, 24], minval=0.0, maxval=1.0))/2
            td_targets =  tf.expand_dims(tf.cast(rewards, tf.float32), -1) + self.GAMMA * tf.vectorized_map(self.linear_spline, (temp_alpha, target_gamma, target_beta, target_delta))
            
            critic_quantile_dl = tf.vectorized_map(self.knot_value, (gamma, beta, delta))
            crps = tf.math.reduce_mean(tf.vectorized_map(self.crps, (td_targets, gamma, beta, delta, critic_quantile_dl)))
                            
        grad_phi = tape.gradient(crps, self.critic.weights)
        self.optimizer_phi.apply_gradients(zip(grad_phi, self.critic.weights))
    
    def train(self, MAX_EPISODE_NUM):

        for ep in range(MAX_EPISODE_NUM):

            # 에피소드 초기화
            time, episode_reward, done = 0, 0, False
            # 환경 초기화 및 초기 상태 관측
            state = self.env.reset()
            
            while not done:
                
                action = self.actor(tf.convert_to_tensor([state], dtype=tf.float32))
                action = action.numpy()[0]
                noise = self.gaussian_noise()
                # 행동 범위 클리핑
                action = np.clip(action + self.DELTA * noise, -self.action_bound, self.action_bound)
                # 다음 상태, 보상 관측
                next_state, reward, done, _ = self.env.step(action)
                # 학습용 보상 설정
                train_reward = (reward + 8) / 8 * 10
                # 리플레이 버퍼에 저장
                self.buffer.add_buffer(state, action, train_reward, next_state, done)

                if self.buffer.buffer_count() > 1000:
                    
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.BATCH_SIZE)
                    states = tf.cast(states, dtype=tf.float32)
                    next_states = tf.cast(next_states, dtype=tf.float32)
                    dones = tf.cast(dones, tf.float32)
                    #grad_theta, action_grad, grad_theta_ = self.actor_learn(states)
                    grad_theta = self.actor_learn(states)
                    
                    self.critic_learn(states, actions, rewards, next_states)
                    self.update_target_network()
                    
                state = next_state
                episode_reward += reward
                time += 1
                
            self.save_epi_reward.append(round(episode_reward, 2))
            
            # self.GAMMA = 0.3 * (1.00001)**(ep/10)                

            if ((ep+1) % 50) == 0:
                print(grad_theta)
                # print(action_grad)
                # print(grad_theta_)
                print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', round(episode_reward, 2))          
                
            if ((ep+1) % 1000) == 0:
                window_size = 100
                rewards_series = pd.Series(self.save_epi_reward)
                windows = rewards_series.rolling(window_size)
                moving_averages = windows.mean()
                moving_averages_list = moving_averages.tolist()
                without_nans = moving_averages_list[window_size - 1:]
                plt.plot(without_nans)
   
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()
        

env = gym.make("Pendulum-v1")

agent = PDPGagent(env)

MAX_EPISODE_NUM = 400

agent.train(MAX_EPISODE_NUM)

agent.plot_result()
