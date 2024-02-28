# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.distributions as td
from torch.nn import functional as F
import gym
from gym import spaces
import numpy as np
from typing import NamedTuple
import warnings
from matplotlib import pyplot as plt
import os
import csv

from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.preprocessing import (
    get_obs_shape, get_action_dim
)


class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    
    def __init__(self,
                 buffer_size,
                 observation_space,
                 action_space,
                 gamma=0.90,
                 device='cpu'):
        
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.device = device
        self.obs_shape = get_obs_shape(self.observation_space)
        self.action_dim = get_action_dim(self.action_space)
        
        self.reset()
        
    def reset(self):
        
        self.observations = np.zeros(
            (self.buffer_size,) + self.obs_shape, dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size, self.action_dim), dtype=np.float32
        )
        self.rewards = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.episode_starts = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.values = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.log_probs = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        self.advantages = np.zeros(
            (self.buffer_size,), dtype=np.float32
        )
        
        self.full = False
        self.pos = 0
        
    def compute_returns_and_advantage(self, last_value, done):
        
        last_value = last_value.clone().cpu().numpy().flatten()
        
        discounted_reward = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_value = self.values[step + 1]
            discounted_reward = self.rewards[step] + \
                self.gamma * discounted_reward * next_non_terminal
            self.advantages[step] = discounted_reward - self.values[step]
        self.returns = self.advantages + self.values
        
    def add(self, obs, action, reward, episode_start, value, log_prob):
        
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)
        
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((1,) + self.obs_shape)
            
        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            
    def get(self, batch_size=None):
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size)

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds):
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
    
    def to_torch(self, array, copy=True):
        if copy:
            return torch.tensor(array).to(self.device)
        return torch.as_tensor(array).to(self.device)


class PolicyNetwork(nn.Module):
    """Base class for stochastic policy networks."""

    def __init__(self):
        super().__init__()

    def forward(self, state):
        """Take state as input, then output the parameters of the policy."""

        raise NotImplemented("forward not implemented.")

    def sample(self, state):
        """
        Sample an action based on the model parameters given the current state.
        """

        raise NotImplemented("sample not implemented.")

    def log_probs(self, obs, actions):
        """
        Return log probabilities for each state-action pair.
        """

        raise NotImplemented("log_probs not implemented.")

    def entropy(self, obs):
        """
        Return entropy of the policy for each state.
        """

        raise NotImplemented("entropy not implemented.")


class GaussianPolicyBase(PolicyNetwork):
    """
    Base class for Gaussian policy.

    Desired network needs to be implemented.
    """

    def __init__(self, action_dim):

        super().__init__()

        self.action_dim = action_dim

    def _get_covs(self, log_stds):
        batch_size = log_stds.shape[0]
        stds = log_stds.exp().reshape(batch_size, 1, 1)
        covs = stds * torch.eye(self.action_dim).repeat(batch_size, 1, 1)
        return covs

    def sample(self, obs, no_log_prob=False):
        mean, log_std = self.forward(obs)
        cov = log_std.exp() * torch.eye(self.action_dim)
        dist = td.MultivariateNormal(mean, cov)
        action = dist.rsample()
        return action if no_log_prob else (action, dist.log_prob(action))

    def log_probs(self, obs, actions):
        means, log_stds = self.forward(obs)
        covs = self._get_covs(log_stds)
        dists = td.MultivariateNormal(means, covs)
        return dists.log_prob(actions)

    def entropy(self, obs):
        means, log_stds = self.forward(obs)
        covs = self._get_covs(log_stds)
        dists = td.MultivariateNormal(means, covs)
        return dists.entropy()


class GaussianPolicy(GaussianPolicyBase):
    """
    Gaussian policy using a two-layer, two-headed MLP with ReLU activation.
    """

    def __init__(self, obs_dim, action_dim,
                 min_action_val=-20.0 * np.array([1, 1]),
                 max_action_val=20.0 * np.array([1, 1]),
                 hidden_layer1_size=64,
                 hidden_layer2_size=64):

        super().__init__(action_dim)

        self.base_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_layer1_size),
            nn.ReLU(),
            nn.Linear(hidden_layer1_size, hidden_layer2_size),
            nn.ReLU(),
        )

        self.mean_net = nn.Sequential(
            nn.Linear(hidden_layer2_size, action_dim),
            nn.Hardtanh(min_action_val[0], max_action_val[0]),
            nn.Hardtanh(min_action_val[1], max_action_val[1])
        )

        self.log_std_net = nn.Sequential(
            nn.Linear(hidden_layer2_size, 1),
        )

    def forward(self, obs):
        x = self.base_net(obs)
        mean = self.mean_net(x)
        log_std = self.log_std_net(x)
        return mean, log_std


class PPOBase:
    def __init__(self,
                 env,
                 policy,
                 value_function,
                 policy_lr,
                 value_lr,
                 entropy_coef=0.0,
                 clip_range=0.2,
                 n_epochs=10,
                 batch_size=64,
                 weight_decay=0.0,
                 gamma=0.99,
                 buffer_size=2048,
                 enable_cuda=True,
                 policy_optimizer=torch.optim.Adam,
                 value_optimizer=torch.optim.Adam,
                 grad_clip_radius=None):

        warnings.warn('This PPO implementation currently contains hacks for ' + \
                      'returning information about CBF-related safety.')

        self.env = env
        self.pi = policy
        self.v = value_function
        self.entropy_coef = entropy_coef
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.__cuda_enabled = enable_cuda
        self.enable_cuda(self.__cuda_enabled, warn=False)
        # NOTE: self.device is defined when self.enable_cuda is called!

        self.pi_optim = policy_optimizer(self.pi.parameters(),
                                         lr=policy_lr,
                                         weight_decay=weight_decay)
        self.v_optim = value_optimizer(self.v.parameters(), lr=value_lr)
        self.grad_clip_radius = grad_clip_radius

        self.rollout_buffer = RolloutBuffer(
            buffer_size,
            env.observation_space,
            env.action_space,
            device=self.device,
            gamma=gamma
        )

    @property
    def cuda_enabled(self):
        return self.__cuda_enabled

    def enable_cuda(self, enable_cuda=True, warn=True):
        """Enable or disable cuda and update models."""
        
        if warn:
            warnings.warn("Converting models between 'cpu' and 'cuda' after "
                          "initializing optimizers can give errors when using "
                          "optimizers other than SGD or Adam!")
        
        self.__cuda_enabled = enable_cuda
        self.device = torch.device(
                'cuda' if torch.cuda.is_available() and self.__cuda_enabled \
                else 'cpu')
        self.pi.to(self.device)
        self.v.to(self.device)
        
    def load_models(self, filename, enable_cuda=True, continue_training=True):
        """
        Load policy and value functions. Copy them to target functions.

        This method is for evaluation only. Use load_checkpoint to continue
        training.
        """
        
        models = torch.load(filename)

        self.pi.load_state_dict(models['pi_state_dict'])
        self.v.load_state_dict(models['v_state_dict'])

        self.pi.eval()
        self.v.eval()
            
        self.enable_cuda(enable_cuda, warn=False)
        
    def save_checkpoint(self, filename):
        """Save state_dicts of models and optimizers."""
        
        torch.save({
                'using_cuda': self.__cuda_enabled,
                'pi_state_dict': self.pi.state_dict(),
                'pi_optimizer_state_dict': self.pi_optim.state_dict(),
                'v_state_dict': self.v.state_dict(),
                'v_optimizer_state_dict': self.v_optim.state_dict(),
        }, filename)
    
    def load_checkpoint(self, filename, continue_training=True):
        """Load state_dicts for models and optimizers."""
        
        checkpoint = torch.load(filename)
        
        self.pi.load_state_dict(checkpoint['pi_state_dict'])
        self.pi_optim.load_state_dict(checkpoint['pi_optimizer_state_dict'])
        self.v.load_state_dict(models['v_state_dict'])
        self.v_optim.load_state_dict(models['v_optimizer_state_dict'])
        
        if continue_training:
            self.pi.train()
            self.v.train()
        else:
            self.pi.eval()
            self.v.eval()
        
        self.enable_cuda(checkpoint['using_cuda'], warn=False)

    def collect_rollout(self, env, rollout_length):
        """
        Perform a rollout and fill the rollout buffer.
        """

        self._last_obs = env.reset()
        self._last_episode_start = np.zeros(1)
        n_steps = 0
        self.rollout_buffer.reset()

        num_unsafe_steps = 0
        x_t=[]
        y_t=[]
        
        local_flag_done = False
        while n_steps < rollout_length:
            action_dim=get_action_dim(env.action_space)

            
            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device).float()
                action, log_prob = self.pi.sample(obs_tensor, action_dim)
                value = self.v(obs_tensor)
            action = action.cpu().numpy()

            # Rescale and perform action
            clipped_action = action
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_action = np.clip(action, self.env.action_space.low,
                                         self.env.action_space.high)
            elif isinstance(self.env.action_space, gym.spaces.Discrete):
                clipped_action = int(clipped_action)

            new_obs, reward, done, info = env.step(clipped_action)


            if abs(new_obs[0]-env.obstacle[0])<0.1 and abs(new_obs[1]-env.obstacle[1])<0.1:
                print("crash")

            n_steps += 1

            if isinstance(self.env.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                action = action.reshape(-1, 1)

            self.rollout_buffer.add(self._last_obs, action, reward,
                                    self._last_episode_start, value, log_prob)
            self._last_obs = new_obs.flatten()
            self._last_episode_start = done

            if done == 0 and local_flag_done == False:
                x_t.append(new_obs[0])
                y_t.append(new_obs[1])
                
            elif done == 1 and n_steps>1:
                local_flag_done = True
    
        plt.xlim(np.double(env.min_x),np.double(env.max_x))
        plt.ylim(np.double(env.min_y),np.double(env.max_y))
        plt.xlabel('X axis')
        plt.ylabel('Y-axis')
        plt.plot(x_t,y_t)
        plt.plot(env.goal[0],env.goal[1],marker='o',color='red')
        plt.plot(env.obstacle[0],env.obstacle[1],marker='*',color='black')

        def f(x, y, xa, yb, a, b):
            return (x - xa)**4/a**4 + (y - yb)**4/b**4

        # Define the point around which to plot
        xa, yb = env.obstacle[0], env.obstacle[1]

        # Define the range of x and y values to plot
        x_vals = np.linspace(xa - env.a_d, xa + env.a_d, 100)
        y_vals = np.linspace(yb - env.b_d, yb + env.b_d, 100)

        # Create a grid of x and y values
        X, Y = np.meshgrid(x_vals, y_vals)

        # Evaluate the function at each point in the grid
        Z = f(X, Y, xa, yb, env.a_d, env.b_d)

        # Plot the function as a contour plot
        

        folder_name_main = f"{{{env.date}}}"
        os.makedirs(folder_name_main, exist_ok=True)
        ##Change the current working directory to the newly created folder
        os.chdir(folder_name_main)
        
        folder_name = f"{{run={env.run}_dt={env.dt}_device={env.device_run}_cbf={env.env_cbf}_roll={rollout_length}}}"
        os.makedirs(folder_name, exist_ok=True)
        ##Change the current working directory to the newly created folder
        os.chdir(folder_name)

        
        folder_name_1 = f"{{lr={env.lr}_entr={env.entropy}_umin={env.umin[0]}_umax={env.umax[0]}_lyr=batch={env.layer_size}}}"
        os.makedirs(folder_name_1, exist_ok=True)
        os.chdir(folder_name_1)

        if (env.episodes)%1 == 0:
            plt.savefig(f"ep={env.episodes}.png")        
            with open(f"episode={env.episodes}.csv", 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(x_t)
                writer.writerow(y_t)
        plt.contour(X, Y, Z, levels=[env.safety_dist])

        
        # Return to the original working directory (optional)
        os.chdir('..')
        os.chdir('..')
        os.chdir('..')
        
       
        self.rollout_buffer.compute_returns_and_advantage(last_value=value,
                                                          done=done)
        
        safety_rate = 100 * (1 - num_unsafe_steps / rollout_length)

        return np.sum(self.rollout_buffer.rewards), safety_rate    

    def train(self):
        """
        Train on the current rollout buffer.
        """        
        for epoch in range(self.n_epochs):

            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):

                actions = rollout_data.actions
                obs = rollout_data.observations
                values = self.v(obs).flatten()
                try:
                    log_probs = self.pi.log_probs(obs, actions)
                except:
                    print(self.pi.log_probs(obs, actions))
                    import pdb; pdb.set_trace()

                entropies = self.pi.entropy(obs)
                if log_probs.device!=actions.device:
                    log_probs=log_probs.to('cuda:0')
                    entropies=entropies.to('cuda:0')
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_probs - rollout_data.old_log_prob)


                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio,
                                                         1 - self.clip_range,
                                                         1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean() - \
                        self.entropy_coef * entropies.mean()

                self.pi_optim.zero_grad()
                policy_loss.backward()
                # Clip grad norm
                if self.grad_clip_radius is not None:
                    torch.nn.utils.clip_grad_norm_(self.pi.parameters(),
                                                   self.grad_clip_radius)
                self.pi_optim.step()

                value_loss = F.mse_loss(rollout_data.returns, values)

                self.v_optim.zero_grad()
                value_loss.backward()
                # Clip grad norm
                if self.grad_clip_radius is not None:
                    torch.nn.utils.clip_grad_norm_(self.v.parameters(),
                                                   self.grad_clip_radius)
                self.v_optim.step()
