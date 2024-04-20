# cbf-constrained_ppo

This repository contains the framework used to conduct the experiments for our paper "Sampling-Based Safe Reinforcement Learning for Nonlinear Dynamical Systems", appearing in _Proceedings of the 27th International Conference on Artificial Intelligence and Statistics (AISTATS)_, 2024. This paper can be found here: https://arxiv.org/abs/2403.04007

In particular, this repo contains an implementation of Control Barrier Function (CBF) constrained policies in `ppo.py` that constructs a Beta policy over the safe control set obtained from the `cbf` function defined in `quad_gym_env.py`, and this policy is then updated using proximal policy optimization defined in `ppo.py`.

Moreover, this repo contains implementation of safe RL policy (in `ppo_proj.pu`) using CBF (in `quad_gym_env_proj.py`) filters. This essentially leads to a projection based safe RL policy.

### Usage

1) Install the packages in `setup.py`
2) For Quadcopter experiments:
    Go to `experiments` directory and select the experiment (e.g., `Testing-projection.py` or `Testing-beta-sampling.py`) that you wish to run
3) For Pendulum Experiments:
    Go to Pendulum directory and run the `Testing-pendulum.py`
5) You'll see plots and rewards arrays being stored in the corresponding experiment folder
