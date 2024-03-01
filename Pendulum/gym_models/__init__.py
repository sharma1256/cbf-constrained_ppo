from gym.envs.registration import register

register(
    id='double_integrator-v0',
    entry_point='gym_models.envs:DoubleIntegrator',
)

register(
    id='inverted_pendulum-v0',
    entry_point='gym_models.envs:InvertedPendulum',
)
