# Control Barrier Function-constrained Proximal Policy Optimization

This repository contains the framework used to conduct the experiments for our paper "Sampling-Based Safe Reinforcement Learning for Nonlinear Dynamical Systems", appearing in _Proceedings of the 27th International Conference on Artificial Intelligence and Statistics (AISTATS)_, 2024. The paper is available [here](https://arxiv.org/abs/2403.04007).

In particular, this repo contains an implementation of control barrier function-(CBF-)constrained policies in `ppo.py` that constructs a Beta policy over the safe control set obtained from the `cbf` function defined in `quad_gym_env.py`, and this policy is then updated using proximal policy optimization algorithm defined in `ppo.py`, which was adapted from [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/). In addition, we have included an implementation of projection-based safe RL policies in `ppo_proj.py` using the CBFs defined in `quad_gym_env_proj.py` to obtain safety constraints. This essentially leads to a projection based safe RL policy like that proposed in [Cheng et al., 2019](https://cdn.aaai.org/ojs/4213/4213-13-7267-1-10-20190705.pdf).

Some of the dynamical components involved in our safe quadcopter gym environment are adapted from the repo: https://github.com/hocherie/cbf_quadrotor

### Usage

1) To install, first set up your preferred virtual environment, then do `pip install -e .`
2) For Quadcopter experiments:
    Go to `experiments` directory and select the experiment (e.g., `Testing-projection.py` or `Testing-beta-sampling.py`) that you wish to run
3) For Pendulum Experiments:
    Go to Pendulum directory and run `Testing-pendulum.py`
5) You'll see plots and rewards arrays being stored in the corresponding experiment folder
