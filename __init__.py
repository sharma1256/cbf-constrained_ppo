from gym.envs.registration import register

register(
    id='quad_gym_env',
    entry_point='gym_models.envs:QuadDynamics',
)

register(
    id='proj_quad_gym_env',
    entry_point='gym_models.envs:QuadDynamicsProj',
)
