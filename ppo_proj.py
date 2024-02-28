# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:26:22 2023

@author: User
"""

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
from matplotlib.animation import FuncAnimation
import datetime
import os
import math
import csv

from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.preprocessing import (
    get_obs_shape, get_action_dim
)


#from agents.TruncatedNormal import TruncatedNormal as tn

from wesutils import two_layer_net

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


# class TruncatedNormalPolicyBase(PolicyNetwork):
#     """
#     Base class for TruncatedNormal policy. Action dimension must be 1.
#     Uses a function (e.g., a CBF) to convert observations into bounds for
#     the TruncatedNormal distribution.

#     Desired network needs to be implemented.
#     """

#     def __init__(self, constraint_fn):

#         super().__init__()

#         self.constraint_fn = self._vectorize_f(constraint_fn)

#     def _get_dist(self, obs):
#         mean, log_std = self.forward(obs)
#         std = log_std.exp()
#         lb, ub = self.constraint_fn(obs)
#         dist = tn(mean, std, lb, ub)

#         return dist

#     def sample(self, obs, no_log_prob=False):
#         dist = self._get_dist(obs)
#         lb = dist.scale * dist.a + dist.loc
#         ub = dist.scale * dist.b + dist.loc
#         action = dist.rsample()

#         for _ in range(100):
#             if (action > lb) and (action < ub):
#                 break
#         if (action < lb) or (action > ub):
#             print("OOPS! Resampling...")
#             action = lb + (ub - lb) * torch.rand(size=(1, 1), requires_grad=True)

#         try:
#             log_prob = dist.log_prob(action)
#         except ValueError:
#             import pdb; pdb.set_trace()

#         return action if no_log_prob else (action, log_prob)

#     def log_probs(self, obs, actions):
#         dists = self._get_dist(obs)
#         return dists.log_prob(actions).flatten()

#     def entropy(self, obs):
#         dists = self._get_dist(obs)
#         return dists.entropy

#     def _vectorize_f(self, f):
#         """
#         Converts a function f defined on 1D numpy arrays and outputting pairs of
#         scalars into a vectorized function accepting batches of
#         torch tensorized arrays and output pairs of torch tensors.
#         """

#         def vectorized_cbf(obs):

#             obs = obs.cpu().detach().numpy()
#             lbs, ubs = [], []

#             if len(obs.shape) == 1:
#                 batch_size = 1
#                 lb, ub = f(obs)
#                 lbs.append(lb)
#                 ubs.append(ub)

#             else:
#                 batch_size = obs.shape[0]
#                 for i in range(batch_size):
#                     lb, ub = f(obs[i])
#                     lbs.append(lb)
#                     ubs.append(ub)

#             lbs = torch.FloatTensor(lbs).reshape(batch_size, 1)
#             ubs = torch.FloatTensor(ubs).reshape(batch_size, 1)
            
#             return lbs, ubs

#         return vectorized_cbf


# class TruncatedNormalPolicy(TruncatedNormalPolicyBase):
#     """
#     TruncatedNormal policy using a two-layer, two-headed MLP with ReLU
#     activation. Action dimension must be 1.
#     """

#     def __init__(self, obs_dim, constraint_fn,
#                  hidden_layer1_size=64,
#                  hidden_layer2_size=64,
#                  mean_min=-np.inf,
#                  mean_max=np.inf,
#                  log_std_lb=-10,
#                  log_std_ub=3):

#         super().__init__(constraint_fn)

#         self.base_net = nn.Sequential(
#             nn.Linear(obs_dim, hidden_layer1_size),
#             nn.ReLU(),
#             nn.Linear(hidden_layer1_size, hidden_layer2_size),
#             nn.ReLU(),
#         )

#         self.mean_net = nn.Sequential(
#             nn.Linear(hidden_layer2_size, 1),
#             # nn.Hardtanh(min_val=mean_min, max_val=mean_max)
#         )

#         self.log_std_net = nn.Sequential(
#             nn.Linear(hidden_layer2_size, 1),
#             # nn.Hardtanh(min_val=log_std_lb, max_val=log_std_ub)
#         )

#         def init_weights(m):
#             if isinstance(m, nn.Linear):
#                 torch.nn.init.normal_(m.weight, std=1.0)

#         self.base_net.apply(init_weights)
#         self.mean_net.apply(init_weights)
#         self.log_std_net.apply(init_weights)

#     def forward(self, obs):
#         x = self.base_net(obs)
#         mean = self.mean_net(x)
#         log_std = self.log_std_net(x)
#         return mean, log_std


class BetaPolicyBase(PolicyNetwork):
    """
    Base class for Beta policy.

    Desired network needs to be implemented.
    """

    def __init__(self, constraint_fn, action_dim, enable_cuda=False):

        super().__init__()

        self.device = torch.device(
                'cuda' if torch.cuda.is_available() and enable_cuda \
                else 'cpu')
        self.constraint_fn = self._vectorize_f(constraint_fn, action_dim)
        self.action_dim = action_dim

    def _vectorize_f(self, f, action_dim): #--vipul :added action_dim
        """
        Converts a function f defined on 1D numpy arrays and outputting pairs of
        scalars into a vectorized function accepting batches of
        torch tensorized arrays and output pairs of torch tensors.
        """

        def vectorized_f(obs, action_dim): #--vipul :added action_dim

            obs = obs.cpu().detach().numpy()

            if len(obs.shape) == 1:  # check to see if obs is a batch or single obs
                batch_size = 1
                lbs, ubs = f(obs)

            else:
                batch_size = obs.shape[0]
                lbs = np.zeros([batch_size, self.action_dim])
                ubs = np.zeros([batch_size, self.action_dim])
                for i in range(batch_size):
                    lbs[i], ubs[i] = f(obs[i])

            lbs = torch.FloatTensor(lbs).reshape(batch_size, self.action_dim)
            ubs = torch.FloatTensor(ubs).reshape(batch_size, self.action_dim)
            lbs = lbs.to(self.device)
            ubs = ubs.to(self.device)
            
            return lbs, ubs

        return vectorized_f

    def sample(self, obs, action_dim, no_log_prob=False):
        """
        Sample from independent Beta distributions across each action_dim.
        """

        assert len(obs.shape) == 1, 'obs must be a flat array'

        alphas, betas = self.forward(obs)
        alphas, betas = torch.flatten(alphas), torch.flatten(betas)
        dists = [
            td.Beta(alpha, beta) for alpha, beta in zip(alphas, betas)
        ]
        action_along_dims = [dist.rsample() for dist in dists]
        action = torch.tensor(action_along_dims, requires_grad=True)
        log_prob = torch.sum(torch.tensor([
            dist.log_prob(a) for dist, a in zip(dists, action_along_dims)
        ], requires_grad=True))
        lb, ub = self.constraint_fn(obs, action_dim)
        action = lb + (ub - lb) * action
        return action if no_log_prob else (action, log_prob)

    def log_probs(self, obs, actions, action_dim):
        alphas_arr, betas_arr = self.forward(obs)
        dists = []
        #import pdb; pdb.set_trace()
        
        #08.28.2023 -vipul making last attempts
        alphas_arr_1 = alphas_arr[:,0]
        alphas_arr_2 = alphas_arr[:,1]
        betas_arr_1 = betas_arr[:,0]
        betas_arr_2 = betas_arr[:,1]
        try:
            dists_1 = td.Beta(alphas_arr_1, betas_arr_1)
        except:
            import pdb; pdb.set_trace()
            
        try:
             dists_2 = td.Beta(alphas_arr_2, betas_arr_2)
        except:
            import pdb; pdb.set_trace()
        #08.23.2023 -vipul done making last attempts
        
        for i in range(alphas_arr.shape[0]):
            alphas = alphas_arr[i]
            betas = betas_arr[i]
            dists.append([
                td.Beta(alpha, beta) for alpha, beta in zip(alphas, betas)
            ])

        lbs, ubs = self.constraint_fn(obs, action_dim)
        if lbs.device!=actions.device:
            lbs = lbs.to('cuda:0')
            ubs = ubs.to('cuda:0')
        actions = (actions - lbs) / (ubs - lbs) 
        actions = actions.clip(0, 1)

        log_probs = []
        for action, action_dists in zip(actions, dists):
            log_probs.append(
                torch.sum(torch.tensor([
                    dim_dist.log_prob(dim_action) \
                        for dim_dist, dim_action in zip(action_dists, action)
                ], requires_grad=True))
            )
        log_probs = torch.tensor(log_probs, requires_grad=True)
        
        #08.28.2023 -vipul making last attempts
        return_new = dists_1.log_prob(actions[:,0]).flatten() + dists_2.log_prob(actions[:,1]).flatten()
        #08.23.2023 -vipul done making last attempts
        
        #return log_probs #--original return  08.28.2023
        
        return return_new #--vipul's return  08.28.2023

    def entropy(self, obs):
        """
        Returns sum of entropies along each independent action dimension.
        """
        alphas_arr, betas_arr = self.forward(obs)
        dists = []
        for i in range(alphas_arr.shape[0]):
            alphas = alphas_arr[i]
            betas = betas_arr[i]
            dists.append([
                td.Beta(alpha, beta) for alpha, beta in zip(alphas, betas)
            ])
        entropies = torch.tensor(
            [torch.sum(torch.tensor([dist.entropy() for dist in dist_list])) \
             for dist_list in dists]
        )
        return entropies
        
        # # TODO: make this consistent with sample & log_probs defined above
        # alphas, betas = self.forward(obs)
        # dists = td.Beta(alphas, betas)
        # return dists.entropy()


### NOTE: this is the BetaPolicyBase giving us the Dirichlet error
# class BetaPolicyBase(PolicyNetwork):
#     """
#     Base class for Beta policy.
# 
#     Desired network needs to be implemented.
#     """
# 
#     def __init__(self, constraint_fn, action_dim):
# 
#         super().__init__()
# 
#         self.constraint_fn = self._vectorize_f(constraint_fn, action_dim)
#         self.action_dim = action_dim
#         #assert self.action_dim == 1, 'Action dimension must be 1'  #-Line commented out--vipul
# 
#     def _vectorize_f(self, f, action_dim): #--vipul :added action_dim
#         """
#         Converts a function f defined on 1D numpy arrays and outputting pairs of
#         scalars into a vectorized function accepting batches of
#         torch tensorized arrays and output pairs of torch tensors.
#         """
# 
#         def vectorized_f(obs, action_dim): #--vipul :added action_dim
# 
#             obs = obs.cpu().detach().numpy()
# 
#             if len(obs.shape) == 1:  # check to see if obs is a batch or single obs
#                 batch_size = 1
#                 lbs, ubs = f(obs)
# 
#             else:
#                 batch_size = obs.shape[0]
#                 lbs = np.zeros([batch_size, self.action_dim])
#                 ubs = np.zeros([batch_size, self.action_dim])
#                 for i in range(batch_size):
#                     lbs[i], ubs[i] = f(obs[i])
# 
#             lbs = torch.FloatTensor(lbs).reshape(batch_size, self.action_dim)
#             ubs = torch.FloatTensor(ubs).reshape(batch_size, self.action_dim)
#             
#             return lbs, ubs
# 
#         return vectorized_f
# 
#     def sample(self, obs, action_dim, no_log_prob=False): #sample from interval is taken with a beta distribution -vipul
#         #sample definition changed to take in argument action_dim --vipul
#         alpha, beta = self.forward(obs)
#         dist = td.Beta(alpha, beta)
#         action = dist.rsample()
#         #log_prob = dist.log_prob(action) #original code
#         log_prob_vec = dist.log_prob(action) #vipul
#         log_prob = sum(log_prob_vec)
#         lb, ub = self.constraint_fn(obs, action_dim)
#         action = lb + (ub - lb) * action  #need to be changed? --vipul
#         #code change nedded! -vipul
#         return action if no_log_prob else (action, log_prob)
# 
#     def log_probs(self, obs, actions, action_dim): #function modified 
#         alphas, betas = self.forward(obs)
#         dists = td.Beta(alphas, betas)
#         lbs, ubs = self.constraint_fn(obs, action_dim)
#         actions = (actions - lbs) / (ubs - lbs) 
#         #need to be changed? --vipul
#         #yes, change needed!
#         return sum(dists.log_prob(actions).flatten())
# 
#     def entropy(self, obs):
#         alphas, betas = self.forward(obs)
#         dists = td.Beta(alphas, betas)
#         return dists.entropy()
###


class BetaPolicy(BetaPolicyBase):
    """
    Beta policy using a two-layer, two-headed MLP with ReLU activation.
    """

    def __init__(self, obs_dim, constraint_fn, action_dim, #vipul : action_dim=1 replaced by just action_dim
                 hidden_layer1_size=64,
                 hidden_layer2_size=64):

        super().__init__(constraint_fn, action_dim=action_dim)

        self.base_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_layer1_size),
            nn.Tanh(),
            nn.Linear(hidden_layer1_size, hidden_layer2_size),
            nn.Tanh(),
        )

        self.alpha_net = nn.Sequential(
            nn.Linear(hidden_layer2_size, action_dim), #vipul : 1 replaced with action_dim
            nn.Softplus(),
        )

        self.beta_net = nn.Sequential(
            nn.Linear(hidden_layer2_size, action_dim), #vipul : 1 replaced with action_dim
            nn.Softplus(),
        )

    def forward(self, obs):

        x = self.base_net(obs)
        # if math.isnan(x[0].item()):
        #     import pdb; pdb.set_trace()
        alpha = 1.0 + self.alpha_net(x) #is there any change needed? --vipul
        beta = 1.0 + self.beta_net(x)   #is there any change needed? --vipul

        return alpha, beta


class CategoricalPolicy(PolicyNetwork):
    """
    Base class for categorical policy.

    Desired network needs to be implemented.
    """

    def __init__(self, num_actions):

        super().__init__()

        self.num_actions = num_actions

    def sample(self, obs, no_log_prob=False):
        logits = self.forward(obs)
        dist = td.Categorical(logits=logits)
        action = dist.sample(sample_shape=torch.tensor([1]))
        return action if no_log_prob else (action, dist.log_prob(action))

    def log_probs(self, obs, actions):
        dists = td.Categorical(logits=self.forward(obs))
        return dists.log_prob(actions.flatten())

    def entropy(self, obs):
        dists = td.Categorical(logits=self.forward(obs))
        return dists.entropy()


class CategoricalPolicyTwoLayer(CategoricalPolicy):
    """
    Categorical policy using a fully connected two-layer network.
    """

    def __init__(self, state_dim, num_actions,
                 hidden_layer1_size=64,
                 hidden_layer2_size=64,
                 init_std=0.001):

        super().__init__(num_actions)

        self.init_std = init_std

        self.linear1 = nn.Linear(state_dim, hidden_layer1_size)
        self.linear2 = nn.Linear(hidden_layer1_size, hidden_layer2_size)
        self.linear3 = nn.Linear(hidden_layer2_size, num_actions)
        nn.init.normal_(self.linear1.weight, std=init_std)
        nn.init.normal_(self.linear2.weight, std=init_std)
        nn.init.normal_(self.linear3.weight, std=init_std)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        output = self.linear3(x)
        return output


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

            #vipul--- I removed these lines for now!
            #as they are pendulum specific
            # if env._state[0] < env.theta_safety_bounds[0] - 1e-8 or \
            #    env._state[0] > env.theta_safety_bounds[1] + 1e-8:
            #     num_unsafe_steps += 1
            
            action_dim=get_action_dim(env.action_space)
            #action_dim added --vipul
            
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

            #adding the animation code here --vipul

            if abs(new_obs[0]-env.obstacle[0])<0.1 and abs(new_obs[1]-env.obstacle[1])<0.1:
                print("crash")
                #break

            n_steps += 1

            if isinstance(self.env.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                action = action.reshape(-1, 1)

            self.rollout_buffer.add(self._last_obs, action, reward,
                                    self._last_episode_start, value, log_prob)
            self._last_obs = new_obs.flatten()
            self._last_episode_start = done

            # if n_steps == rollout_length:
            #     env.reset()
            if done == 0 and local_flag_done == False:
                x_t.append(new_obs[0])
                y_t.append(new_obs[1])
                
            elif done == 1 and n_steps>1:
                local_flag_done = True




        #code for plotting the quadrotor trajectory in an episode
        # plt.xlim(-200,200)
        # plt.ylim(-200,200)
        # plt.plot(x_t, y_t, color = 'red')
        # plt.pause(1)      
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
        
        #ADDED - 09.1.23
        #plt.savefig(f"{env.date}_run={env.run}_device={env.device_run}_cbf={env.env_cbf}_lr={env.lr}_entr={env.entropy}_umin={env.umin[0]}_umax={env.umax[0]}_lyr=batch={env.layer_size}_roll={rollout_length}.png")
        ##Create a folder in the current directory
        folder_name_main = f"{{{env.date}}}"
        os.makedirs(folder_name_main, exist_ok=True)
        ##Change the current working directory to the newly created folder
        os.chdir(folder_name_main)
        
        folder_name = f"{{run={env.run}_dt={env.dt}_device={env.device_run}_cbf={env.env_cbf}_roll={rollout_length}}}"
        os.makedirs(folder_name, exist_ok=True)
        ##Change the current working directory to the newly created folder
        os.chdir(folder_name)
        
        #folder_name_0 = f"{{run={env.run}}}"

        
        folder_name_1 = f"{{lr={env.lr}_entr={env.entropy}_umin={env.umin[0]}_umax={env.umax[0]}_lyr=batch={env.layer_size}}}"
        os.makedirs(folder_name_1, exist_ok=True)
        os.chdir(folder_name_1)
        #ADDED - 09.1.23

        if (env.episodes)%1 == 0:
            #plt.text(10, 12, f"{env.episodes}", fontsize=10, color='blue', ha='right')
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
        
        #quadrotor.plot_step(new_obs, u_hat_acc, state_hist, plot_handle)
        
        # #creating animation -08.31.23
        # fig, ax = plt.subplots()
        # line, = ax.plot([], [], lw=2)
        # ax.contour(X, Y, Z, levels=[env.safety_dist])
        # ax.plot(env.goal[0],env.goal[1],marker='o',color='red')
        # ax.plot(env.obstacle[0],env.obstacle[1],marker='*',color='black')
        # def init():
        #     ax.set_xlim(-1, 14)
        #     ax.set_ylim(-1, 14)
        #     return line,
        # def update(frame):
        #     line.set_data(x_t[frame], y_t[frame])
        #     return line,
        # ani = FuncAnimation(fig, update, frames=len(x_t), init_func=init, blit=True)
        # ani.save('1st_sep_trajectroy_animation.gif', writer='pillow')
        
        self.rollout_buffer.compute_returns_and_advantage(last_value=value,
                                                          done=done)
        
        safety_rate = 100 * (1 - num_unsafe_steps / rollout_length)

        #return np.sum(self.rollout_buffer.rewards) / np.sum(
            #self.rollout_buffer.episode_starts), safety_rate
        return np.sum(self.rollout_buffer.rewards), safety_rate    

    def train(self):
        """
        Train on the current rollout buffer.
        """        
        #action_dim = get_action_dim(self.action_space)
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
                
                # if ratio[0]>1.001:
                #     print(ratio)
                #     import pdb; pdb.set_trace()


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

        # import pdb; pdb.set_trace()

#this definition is copied from stable-baselines3' website and DOES NOT belong to this code
# def obs_as_tensor(
#     obs: Union[np.ndarray, Dict[Union[str, int], np.ndarray]], device: th.device
# ) -> Union[th.Tensor, TensorDict]:
#     """
#     Moves the observation to the given device.

#     :param obs:
#     :param device: PyTorch device
#     :return: PyTorch tensor of the observation on a desired device.
#     """
#     if isinstance(obs, np.ndarray):
#         return th.as_tensor(obs, device=device)
#     elif isinstance(obs, dict):
#         return {key: th.as_tensor(_obs, device=device) for (key, _obs) in obs.items()}
#     else:
#         raise Exception(f"Unrecognized type of observation {type(obs)}")
