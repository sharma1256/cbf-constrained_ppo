from typing import Union
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding



class DoubleIntegrator(gym.Env):
    """
    Description:
    """


    metadata = {'render.modes': ['console']}

    def __init__(self, 
                 tau: float = 1e-2, 
                 initial_state: Union[float, float] = [9.1, 1],
                 safety_bounds: Union[float, float] = [9., 11.],
                 x1_des: float = [10.],
                 max_steps: int = 1000
                 ):
        super(DoubleIntegrator, self).__init__()

        self._tau = tau
        self.initial_state = initial_state
        self.safety_bounds = safety_bounds
        self.x1_des = x1_des
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,1), dtype=np.float64)
        self._state = np.array(self.initial_state)
        self.max_steps = max_steps
        self.count = 0
        self.seed()

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, value: float):
        if value>1e-1:
            print("discretizing time is too high, consider reducing for better results")
        self._tau = value

    @property
    def state(self):
        return self._state.reshape(2,1)

    def seed(self, seed=None):
        # not used
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward(self):
        x1 = self._state[0]
        return (-0.5) * ((x1 - self.x1_des) ** 2)

    def step(self, action: float):
        self.count += 1
        x1, x2 = self._state[0], self._state[1]
        x1_new = x1 + (self._tau * x2) + (0.5 * (self._tau ** 2)) * action
        x2_new = x2 + (self._tau) * action
        self._state = np.array([x1_new, x2_new], dtype=float)

        x1_min, x1_max = self.safety_bounds[0], self.safety_bounds[1]
        done = False
        if (x1_new < x1_min) or (x1_new > x1_max) or (self.count>self.max_steps):
            done = True
        return self.state, self.reward(), done, {}
        
    def reset(self):
        self._state = np.array(self.initial_state)
        self.count = 0
        return self.state
    
    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print("not implemented")

    def close(self):
        pass
    
    def cbf(self, state=None, eta: float = 0.5):
        """
        Calculates CBF constraint set at a given state. Default is
        the current state.
        """

        state = state if state is not None else self._state

        if (eta>1-1e-3) or (eta<1e-5):
            raise ValueError("eta should be inside (0, 1)")
        x1, x2 = state[0], state[1]
        x1_min, x1_max = self.safety_bounds[0], self.safety_bounds[1]
        u_min = (2 / (self._tau ** 2) ) * ( - (self._tau * x2) - eta * (x1 - x1_min))
        u_max = (2 / (self._tau ** 2) ) * ( - (self._tau * x2) + eta * (x1_max - x1))
        if u_min>u_max:
            raise ValueError("Infeasible")
        else:
            return [u_min, u_max]
