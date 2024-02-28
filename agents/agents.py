import torch
import numpy as np
import os
import yaml
from scipy import integrate
from wesutils.utils import GaussianPolicyTwoLayer, two_layer_net
from functools import reduce
from operator import mul
import warnings

class GaussianPolicyCBF(GaussianPolicyTwoLayer):
    """
    Modified version of the standard Gaussian policy for use in CBF-
    constrained settings. Provides access to the pdf of the policy at
    a given state as well as utilities for directly manipulating the
    parameters of the policy. Requires a CBF upon initilization. The
    CBF is assumed to accept torch tensor representations of the state.

    NOTE: This version of the policy assumes that action_dim=1 and that
    the CBF returns intervals.
    """

    def __init__(self, cbf, state_dim, action_dim=1,
                 simple_cov=True,
                 hidden_layer1_size=32,
                 hidden_layer2_size=32,
                 activation='sigmoid',
                 log_std_min=-20, log_std_max=3,
                 weight_init_std=0.0001):

        assert action_dim == 1, "Action dimension must be 1"

        super().__init__(
            state_dim, action_dim,
            simple_cov=simple_cov,
            hidden_layer1_size=hidden_layer1_size,
            hidden_layer2_size=hidden_layer2_size,
            activation=activation,
            log_std_min=log_std_min, log_std_max=log_std_max,
            weight_init_std=weight_init_std
        )

        self.cbf = cbf
        self.param_shape = tuple(p.shape for p in self.parameters())
        self.param_size = sum(reduce(mul, shape) for shape in self.param_shape)

    def to(self, device):
        super().to(device)
        self.device = device

    @property
    def params(self):
        """Get policy model parameters."""
        return torch.cat([p.data.reshape(-1) for p in self.parameters()])

    @params.setter
    def params(self, new_values):
        """Set policy model parameters."""
        assert new_values.size()[0] == self.param_size, "Error"

        index = 0

        for param in self.parameters():
            size = reduce(mul, param.shape)
            block = new_values[index:index+size].reshape(param.shape)
            param.data.copy_(block)
            index += size

    def _numpy_original_pdf(self, state):
        """
        Return numpy version of the pdf of the untruncated policy at the
        state provided.
        """
        
        mean, cov = self.forward(state)
        mean, cov = mean.detach().numpy(), cov.detach().numpy()
        K = np.float_power((2 * np.pi)**len(mean) * np.linalg.det(cov), -0.5)
        inv = np.linalg.inv(cov)

        def pdf(action):
            return K * np.exp(
                -0.5 * (action - mean).dot(inv.dot(action - mean))
            ).flatten()

        return pdf

    def _torch_original_pdf(self, state):
        """
        Return torch version of the pdf of the untruncated policy at the
        state provided. Detach is not called, so all computations herein
        are reflected in the computation graph.
        """

        # import pdb; pdb.set_trace()

        mean, cov = self.forward(state)
        K = torch.float_power((2 * np.pi)**len(mean) * torch.linalg.det(cov), -0.5)
        inv = torch.linalg.inv(cov).squeeze(dim=0)

        def pdf(action):
            return K * torch.exp(-0.5 * torch.matmul(
                action - mean, torch.matmul(inv, action - mean)))

        return pdf

    def get_numpy_pdf(self, state):
        """
        Return the pdf of the original Gaussian pdf truncated to the set C(x).
        """
        
        lb, ub = self.cbf(state=state)
        original_pdf = self._numpy_original_pdf(state)
        normalization = integrate.quad(original_pdf, lb, ub)[0]

        def pdf(action):
            return original_pdf(action) / normalization if lb <= action <= ub \
                else 0

        return pdf

    def sample(self, state, sample_cutoff=100,
               no_log_prob=False, num_log_prob_samples=1000):
        """
        Repeatedly sample from the original Gaussian policy until an action
        lying within the CBF constraint set is generated.

        sample_cutoff specifies the number of times to sample using the
        original Gaussian policy before simply uniformly generating an
        action from the CBF constraint set.
        """

        lb, ub = self.cbf(state=state)
        lb = float(lb)
        ub = float(ub)
        state = torch.FloatTensor(state).reshape(1, len(state)).to(self.device)
        orig_pdf = self._torch_original_pdf(state)

        if not no_log_prob:
            actions = lb + (ub - lb) * torch.rand(
                num_log_prob_samples, self.action_dim, 1, device=self.device
            )
            log_prob = orig_pdf(actions).sum().log() # log pi_theta (C(x) | x)

        for _ in range(sample_cutoff):

            action, orig_log_prob = super().sample(state)

            if lb <= action <= ub:
                log_prob = orig_log_prob - log_prob # log pi^C_theta 
                                                    # = log pi_theta - log pi_theta(C(x) | x)
                return action, log_prob if not no_log_prob else action

        action = lb + (ub - lb) * torch.rand(1, 1)
        orig_log_prob = orig_pdf(action).log()
        return action, (orig_log_prob - log_prob) if not no_log_prob else action


class CBFREINFORCEAgent:
    """
    Agent for training a CBF-constrained version of the classic REINFORCE
    algorithm.
    """

    def __init__(self,
                 ### agent parameters
                 state_dim, action_dim, cbf,
                 policy_lr, discount_factor,
                 num_log_prob_samples=1000,
                 enable_cuda=False,
                 optimizer=torch.optim.Adam,
                 grad_clip_radius=None,
                 ### policy parameters
                 simple_cov=True,
                 hidden_layer1_size=32,
                 hidden_layer2_size=32,
                 activation='relu',
                 log_std_min=-20, log_std_max=3,
                 weight_init_std=0.0001):

        assert action_dim == 1, "Action dimension must be 1 in this version"

        self.pi = GaussianPolicyCBF(
            cbf=cbf, state_dim=state_dim, action_dim=action_dim,
            simple_cov=simple_cov,
            hidden_layer1_size=hidden_layer1_size,
            hidden_layer2_size=hidden_layer2_size,
            activation=activation,
            log_std_min=log_std_min, log_std_max=log_std_max,
            weight_init_std=weight_init_std
        )

        self.gamma = discount_factor

        self.pi_optim = optimizer(self.pi.parameters(), lr=policy_lr)
        self.grad_clip_radius = grad_clip_radius

        self.__cuda_enabled = enable_cuda
        self.enable_cuda(self.__cuda_enabled, warn=False)
        # NOTE: self.device is defined when self.enable_cuda is called!

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
        
    def load_models(self, filename, enable_cuda=True, continue_training=True):
        """Load policy and value functions. Copy them to target functions."""
        
        models = torch.load(filename)

        self.pi.load_state_dict(models['pi_state_dict'])
        
        if continue_training:
            self.pi.train()
        else:
            self.pi.eval()
            
        self.enable_cuda(enable_cuda, warn=False)
        
    def save_checkpoint(self, filename):
        """Save state_dicts of models and optimizers."""
        
        torch.save({
                'using_cuda': self.__cuda_enabled,
                'pi_state_dict': self.pi.state_dict(),
                'pi_optimizer_state_dict': self.pi_optim.state_dict(),
        }, filename)
    
    def load_checkpoint(self, filename, continue_training=True):
        """Load state_dicts for models and optimizers."""
        
        checkpoint = torch.load(filename)
        
        self.pi.load_state_dict(checkpoint['pi_state_dict'])
        self.pi_optim.load_state_dict(checkpoint['pi_optimizer_state_dict'])
        
        if continue_training:
            self.pi.train()
            
        else:
            self.pi.eval()
        
        self.enable_cuda(checkpoint['using_cuda'], warn=False)

    def update(self, env, rollout_length, sample_cutoff=100):
        """
        Perform a single rollout and corresponding gradient update.
        Return the total reward accumulated during the rollout.
        """

        rewards, log_probs = [], []
        num_steps = 0

        state = env.state

        for _ in range(rollout_length):
            action, log_prob = self.pi.sample(state,
                                              sample_cutoff=sample_cutoff)
            state, reward, done, _ = env.step(action.cpu().detach().numpy())
            rewards.append(float(reward))
            log_probs.append(log_prob)
            
            if done:
                break

            num_steps += 1

        G = 0
        pi_loss = 0

        for i in range(len(rewards) - 1, -1, -1):
            G = rewards[i] + self.gamma * G
            pi_loss = pi_loss + (self.gamma ** i) * G * log_probs[i]

        pi_loss = -pi_loss

        self.pi_optim.zero_grad()
        pi_loss.backward()
        if self.grad_clip_radius is not None:
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(),
                                           self.grad_clip_radius)
        self.pi_optim.step()

        return np.mean(rewards)

    def train(self, env, num_episodes, rollout_length,
              output_dir, args_list,
              reset_env=True,
              sample_cutoff=100):
        """
        Train on the environment.
        """

        episode_mean_rewards = []

        for i in range(num_episodes):
            if reset_env:
                env.reset()
            mean_reward = self.update(env, rollout_length,
                                      sample_cutoff=sample_cutoff)
            cbf = [float(elem) for elem in env.cbf(env.state)]
            print(
                f'Episode {i}: moving ave reward {np.mean(episode_mean_rewards[-20:]):.8f}, mean reward {mean_reward:.8f}, C(x) = [{cbf[0]:.2f}, {cbf[1]:.2f}]')
            episode_mean_rewards.append(mean_reward)

        rewards_filename = os.path.join(output_dir, 'episode_rewards')
        np.save(rewards_filename, episode_mean_rewards)

        hyperparams_filename = os.path.join(output_dir, 'hyperparams.yml')
        with open(hyperparams_filename, 'w') as f:
            yaml.dump(args_list, f)



class CBFACAgent:
    """
    Agent for training a CBF-constrained version of the classic actor-critic
    algorithm.
    """

    def __init__(self,
                 ### agent parameters
                 state_dim, action_dim,
                 policy_lr, value_lr, discount_factor,
                 cbf=None,
                 num_log_prob_samples=1000,
                 enable_cuda=False,
                 policy_optimizer=torch.optim.Adam,
                 value_optimizer=torch.optim.Adam,
                 grad_clip_radius=None,
                 ### policy parameters
                 simple_cov=True,
                 policy_hidden_layer1_size=32,
                 policy_hidden_layer2_size=32,
                 policy_activation='relu',
                 log_std_min=-20, log_std_max=3,
                 weight_init_std=0.0001,
                 # value function parameters
                 value_hidden_layer1_size=32,
                 value_hidden_layer2_size=32,
                 value_activation='ReLU'):

        assert action_dim == 1, "Action dimension must be 1 in this version"

        self.pi = GaussianPolicyCBF(
            cbf=cbf, state_dim=state_dim, action_dim=action_dim,
            simple_cov=simple_cov,
            hidden_layer1_size=policy_hidden_layer1_size,
            hidden_layer2_size=policy_hidden_layer2_size,
            activation=policy_activation,
            log_std_min=log_std_min, log_std_max=log_std_max,
            weight_init_std=weight_init_std
        )

        self.v = two_layer_net(
            input_dim=state_dim, output_dim=1,
            hidden_layer1_size=value_hidden_layer1_size,
            hidden_layer2_size=value_hidden_layer2_size,
            activation=value_activation,
        )

        self.gamma = discount_factor

        self.pi_optim = policy_optimizer(self.pi.parameters(), lr=policy_lr)
        self.grad_clip_radius = grad_clip_radius

        self.v_optim = value_optimizer(self.v.parameters(), lr=value_lr)
        self.grad_clip_radius = grad_clip_radius

        self.__cuda_enabled = enable_cuda
        self.enable_cuda(self.__cuda_enabled, warn=False)
        # NOTE: self.device is defined when self.enable_cuda is called!

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
        """Load policy and value functions. Copy them to target functions."""
        
        models = torch.load(filename)

        self.pi.load_state_dict(models['pi_state_dict'])
        self.v.load_state_dict(models['v_state_dict'])
        
        if continue_training:
            self.pi.train()
            self.v.train()
        else:
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
                'v_optimizer_state_dict': self.v_optim.state_dict()
        }, filename)
    
    def load_checkpoint(self, filename, continue_training=True):
        """Load state_dicts for models and optimizers."""
        
        checkpoint = torch.load(filename)
        
        self.pi.load_state_dict(checkpoint['pi_state_dict'])
        self.pi_optim.load_state_dict(checkpoint['pi_optimizer_state_dict'])
        self.v.load_state_dict(checkpoint['v_state_dict'])
        self.v_optim.load_state_dict(checkpoint['v_optimizer_state_dict'])
        
        if continue_training:
            self.pi.train()
            self.v.train()
            
        else:
            self.pi.eval()
            self.v.eval()
        
        self.enable_cuda(checkpoint['using_cuda'], warn=False)

    def update(self, env, episode_length, sample_cutoff=100):
        """
        Perform a single episode and corresponding gradient update.
        Return the total reward accumulated during the rollout.
        """

        states, actions, rewards, next_states, log_probs = [], [], [], [], []

        state = env.state

        for _ in range(episode_length):
            states.append(state)

            action, log_prob = self.pi.sample(state,
                                              sample_cutoff=sample_cutoff)
            actions.append(action)
            log_probs.append(log_prob)
            state, reward, done, _ = env.step(action.cpu().detach().numpy())
            rewards.append(reward)
            next_states.append(state)

            if done:
                break

        next_states.append(env.state)

        v_loss = 0
        pi_loss = 0

        for state, action, reward, next_state, log_prob in zip(
            states, actions, rewards, next_states, log_probs):
            state = torch.FloatTensor(state).reshape(1, len(state))
            next_state = torch.FloatTensor(next_state).reshape(1, len(next_state))
            with torch.no_grad():
                v_target = float(reward) + self.gamma * self.v(next_state)
                td_error = v_target - self.v(state)
            v_loss += (v_target - self.v(state))**2
            pi_loss = pi_loss + td_error * log_prob

        v_loss = v_loss / len(states)
        pi_loss = pi_loss / len(states)
        pi_loss = -pi_loss

        self.pi_optim.zero_grad()
        self.v_optim.zero_grad()

        if self.grad_clip_radius is not None:
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(),
                                           self.grad_clip_radius)
            torch.nn.utils.clip_grad_norm_(self.v.parameters(),
                                           self.grad_clip_radius)

        self.v_optim.step()
        self.pi_optim.step()

        return np.mean(rewards)

    def train(self, env, num_episodes, rollout_length,
              output_dir, args_list,
              reset_env=True,
              sample_cutoff=100):
        """
        Train on the environment.
        """

        episode_mean_rewards = []

        for i in range(num_episodes):
            if reset_env:
                env.reset()
            mean_reward = self.update(env, rollout_length,
                                      sample_cutoff=sample_cutoff)
            episode_mean_rewards.append(mean_reward)
            cbf = [float(elem) for elem in env.cbf(env.state)]
            print(
                f'Episode {i}: moving ave reward {np.mean(episode_mean_rewards[-20:]):.8f}, mean reward {mean_reward:.8f}, C(x) = [{cbf[0]:.2f}, {cbf[1]:.2f}]')

        rewards_filename = os.path.join(output_dir, 'episode_rewards')
        np.save(rewards_filename, episode_mean_rewards)

        hyperparams_filename = os.path.join(output_dir, 'hyperparams.yml')
        with open(hyperparams_filename, 'w') as f:
            yaml.dump(args_list, f)


class VanillaACAgent:
    """
    Agent for training the classic actor-critic algorithm.
    """

    def __init__(self,
                 ### agent parameters
                 state_dim, action_dim,
                 policy_lr, value_lr, discount_factor,
                 enable_cuda=False,
                 policy_optimizer=torch.optim.Adam,
                 value_optimizer=torch.optim.Adam,
                 grad_clip_radius=None,
                 ### policy parameters
                 simple_cov=True,
                 policy_hidden_layer1_size=32,
                 policy_hidden_layer2_size=32,
                 policy_activation='relu',
                 log_std_min=-20, log_std_max=3,
                 weight_init_std=0.0001,
                 # value function parameters
                 value_hidden_layer1_size=32,
                 value_hidden_layer2_size=32,
                 value_activation='ReLU',
                 cbf=None,
                 num_log_prob_samples=None):

        assert action_dim == 1, "Action dimension must be 1 in this version"

        self.pi = GaussianPolicyTwoLayer(
            state_dim=state_dim, action_dim=action_dim,
            simple_cov=simple_cov,
            hidden_layer1_size=policy_hidden_layer1_size,
            hidden_layer2_size=policy_hidden_layer2_size,
            activation=policy_activation,
            log_std_min=log_std_min, log_std_max=log_std_max,
            weight_init_std=weight_init_std
        )

        self.v = two_layer_net(
            input_dim=state_dim, output_dim=1,
            hidden_layer1_size=value_hidden_layer1_size,
            hidden_layer2_size=value_hidden_layer2_size,
            activation=value_activation,
        )

        self.gamma = discount_factor

        self.pi_optim = policy_optimizer(self.pi.parameters(), lr=policy_lr)
        self.grad_clip_radius = grad_clip_radius

        self.v_optim = value_optimizer(self.v.parameters(), lr=value_lr)
        self.grad_clip_radius = grad_clip_radius

        self.__cuda_enabled = enable_cuda
        self.enable_cuda(self.__cuda_enabled, warn=False)
        # NOTE: self.device is defined when self.enable_cuda is called!

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
        """Load policy and value functions. Copy them to target functions."""
        
        models = torch.load(filename)

        self.pi.load_state_dict(models['pi_state_dict'])
        self.v.load_state_dict(models['v_state_dict'])
        
        if continue_training:
            self.pi.train()
            self.v.train()
        else:
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
                'v_optimizer_state_dict': self.v_optim.state_dict()
        }, filename)
    
    def load_checkpoint(self, filename, continue_training=True):
        """Load state_dicts for models and optimizers."""
        
        checkpoint = torch.load(filename)
        
        self.pi.load_state_dict(checkpoint['pi_state_dict'])
        self.pi_optim.load_state_dict(checkpoint['pi_optimizer_state_dict'])
        self.v.load_state_dict(checkpoint['v_state_dict'])
        self.v_optim.load_state_dict(checkpoint['v_optimizer_state_dict'])
        
        if continue_training:
            self.pi.train()
            self.v.train()
            
        else:
            self.pi.eval()
            self.v.eval()
        
        self.enable_cuda(checkpoint['using_cuda'], warn=False)

    def update(self, env, episode_length, sample_cutoff=100):
        """
        Perform a single episode and corresponding gradient update.
        Return the mean reward accumulated during the rollout.
        """

        states, actions, rewards, next_states, log_probs = [], [], [], [], []

        state = env.reset()

        for _ in range(episode_length):
            states.append(state)

            action, log_prob = self.pi.sample(
                torch.FloatTensor(state).reshape(1, len(state)).to(self.device)
            )
            actions.append(action)
            log_probs.append(log_prob)
            state, reward, done, _ = env.step(action.cpu().detach().numpy())
            rewards.append(reward)
            next_states.append(state)
            if done:
                break

        next_states.append(env.state)

        v_loss = 0
        pi_loss = 0

        for state, action, reward, next_state, log_prob in zip(
            states, actions, rewards, next_states, log_probs):
            state = torch.FloatTensor(state).reshape(1, len(state))
            next_state = torch.FloatTensor(next_state).reshape(1, len(next_state))
            with torch.no_grad():
                v_target = float(reward) + self.gamma * self.v(next_state)
                td_error = v_target - self.v(state)
            v_loss += (v_target - self.v(state))**2
            pi_loss = pi_loss + td_error * log_prob

        pi_loss = pi_loss / len(states)
        v_loss = v_loss / len(states)
        pi_loss = -pi_loss

        self.pi_optim.zero_grad()
        self.v_optim.zero_grad()

        if self.grad_clip_radius is not None:
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(),
                                           self.grad_clip_radius)
            torch.nn.utils.clip_grad_norm_(self.v.parameters(),
                                           self.grad_clip_radius)

        self.v_optim.step()
        self.pi_optim.step()

        return np.mean(rewards)

    def train(self, env, num_episodes, rollout_length,
              output_dir, args_list,
              reset_env=True,
              sample_cutoff=100):
        """
        Train on the environment.
        """

        episode_mean_rewards = []

        for i in range(num_episodes):
            if reset_env:
                env.reset()
            mean_reward = self.update(env, rollout_length,
                                      sample_cutoff=sample_cutoff)
            episode_mean_rewards.append(mean_reward)
            print(
                f'Episode {i}: moving ave reward {np.mean(episode_mean_rewards[-20:]):.8f}, mean reward {mean_reward:.8f}')

        # rewards_filename = os.path.join(output_dir, 'episode_rewards')
        # np.save(rewards_filename, episode_mean_rewards)

        # hyperparams_filename = os.path.join(output_dir, 'hyperparams.yml')
        # with open(hyperparams_filename, 'w') as f:
        #     yaml.dump(args_list, f)
