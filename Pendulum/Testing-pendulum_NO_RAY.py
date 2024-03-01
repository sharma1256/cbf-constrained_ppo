# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 12:16:05 2023

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 22:46:55 2023

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 11:25:00 2023

@author: vipul
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 20:16:49 2023

@author: VIPUL
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
from itertools import product
import pickle
import os
import yaml
import time
import shutil
import torch
#import torch.distributions
import wesutils
#from quad_gym_env import QuadDynamics
#from ppo import BetaPolicy
import agents.ppo
import gym_models
import gym_models.envs.pendulum
#from torch import distributions

import pdb


### Hyperparameters...

# ...for the agent
n_episodes = 1000 #1000
#rollout_length = 2048
rollout_length = 2048
buffer_size = rollout_length
policy_lr = 0.0001
value_lr = 0.0001
layer_size = 512
enable_cuda = False
n_epochs = 10
batch_size = 64 #modified from 256
entropy_coef = 0.000001
weight_decay = 0.0
T=5 #storing reward per 100 episodes
# ...for the environment
#dt = 0.03
#max_steps = 700
#umin = -20 * np.array([1, 1])
#umax = 20 * np.array([1, 1])
tau=0.05
theta_safety_bounds=[-1.0, 1.0]
beta_torque_bounds=[-15.0, 15.0]
pi_units1=128
pi_units2=128
v_units1=64
v_units2=64
def train():
    env = gym_models.envs.pendulum.InvertedPendulum(
        tau=tau,
        theta_safety_bounds=theta_safety_bounds,
        torque_bounds=beta_torque_bounds
    )
    
    # pi = BetaPolicy(
    #     10, env.cbf, 2,
    #     hidden_layer1_size=layer_size,
    #     hidden_layer2_size=layer_size,
    # )
    pi = agents.ppo.BetaPolicy(
        3, env.cbf, 1,
        hidden_layer1_size=pi_units1,
        hidden_layer2_size=pi_units2
    )
    # v = wesutils.two_layer_net(
    #     10, 1, layer_size, layer_size
    # )
    v = wesutils.two_layer_net(
        3, 1, v_units1, v_units2
    )
    agent = agents.ppo.PPOBase(
        env, pi, v,
        policy_lr, value_lr,
        buffer_size=buffer_size,
        enable_cuda=enable_cuda,
        n_epochs=n_epochs,
        batch_size=batch_size,
        entropy_coef=entropy_coef,
        weight_decay=weight_decay,
    )
    
    # train and collect data
    rewards, safety_rates = [], [] # TODO: get rid of safety_rates
    for i in range(n_episodes):
        reward, safety_rate = agent.collect_rollout(env, rollout_length)
        agent.train()
        rewards.append(reward)
        safety_rates.append(safety_rate)
        if i%T==0:
            np.save("rewards_sequence_local_duplicate.npy", rewards)
        safety_rates.append(safety_rate)
        
        print(f'Episode {i} return: {reward:.2f}') # does this work?
    
    return {'rewards': rewards,
            'safety_rates': safety_rates}


if __name__ == '__main__':

    start_time = time.time()
    train()
    total_runtime = time.time() - start_time

    print(f'Total runtime: {total_runtime / 60:.1f}m')
