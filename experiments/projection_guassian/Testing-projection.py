# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:38:39 2023

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
import wesutils
import gym_models
import gym_models.envs.pendulum
import agents.ppo
from proj_quad_gym_env import QuadDynamics
from proj_ppo import GaussianPolicy
from proj_ppo import BetaPolicy
import proj_ppo
from datetime import datetime

# Get today's date as a datetime object
today_date = datetime.today()

# Convert the datetime object to a string
today_date_str = today_date.strftime("%Y-%m-%d") 







### Hyperparameters...
n_episodes = 10000 #1000
rollout_length = 320
buffer_size = rollout_length
policy_lr = 0.0004
value_lr = 0.0004
layer_size = 256
enable_cuda = True
n_epochs = 10
batch_size = 256 #modified from 256
entropy_coef = 0.00000001
weight_decay = 0.0
T=1 #storing reward per 100 episodes
# ...for the environment
dt = 0.1
max_steps = 1000
umin = -15.0 * np.array([1, 1])
umax = 15.0 * np.array([1, 1])
episode = 0
cbf = True
device_run = 'projection-interfering-obstacle'
run = 1


def train():
    env = QuadDynamics(
        dt=dt,
        max_steps=max_steps,
        umax=umax,
        umin=umin,
        env_cbf = cbf,
        layer_size = layer_size,
        entropy = entropy_coef,
        lr = policy_lr,
        device_run = device_run,
        date = today_date_str,
        run = run
    )
    
    pi = GaussianPolicy(
        10, 2, umin, umax,
        hidden_layer1_size=layer_size,
        hidden_layer2_size=layer_size,
    )
    # pi = BetaPolicy(
    #     10, env.cbf, 2,
    #     hidden_layer1_size=layer_size,
    #     hidden_layer2_size=layer_size,
    # )
    v = wesutils.two_layer_net(
        10, 1, layer_size, layer_size
    )
    agent = proj_ppo.PPOBase(
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
        env.episodes = i
        reward, safety_rate = agent.collect_rollout(env, rollout_length)
        agent.train()
        rewards.append(reward)
        safety_rates.append(safety_rate)
        if i%T==0:
            folder_name_main = f"{{{env.date}}}"
            os.makedirs(folder_name_main, exist_ok=True)
            ##Change the current working directory to the newly created folder
            os.chdir(folder_name_main)
            
            folder_name = f"{{run={env.run}_dt={env.dt}_device={env.device_run}_cbf={env.env_cbf}_roll={rollout_length}}}"
            os.makedirs(folder_name, exist_ok=True)
            ##Change the current working directory to the newly created folder
            os.chdir(folder_name)
            np.save(f"lr={policy_lr}_ent={entropy_coef}_lyr=batch={layer_size}_roll={rollout_length}.npy", rewards)
            os.chdir('..')
            os.chdir('..')
        safety_rates.append(safety_rate)
        
        print(f'Episode {i} return: {reward:.2f}') # does this work?
    
    return {'rewards': rewards,
            'safety_rates': safety_rates}


if __name__ == '__main__':

    start_time = time.time()
    train()
    total_runtime = time.time() - start_time

    print(f'Total runtime: {total_runtime / 60:.1f}m')
