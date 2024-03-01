from typing import Union
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from os import path
from gym.envs.classic_control import rendering



class InvertedPendulum(gym.Env):
    """
    Description:
    """


    # metadata = {'render.modes': ['console']}
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, 
                 tau: float = 5e-2,
                 m: float = 1,
                 g: float = 9.8,
                 l: float = 1,
                 initial_state: Union[float, float] = [0.0, 0.],
                 theta_safety_bounds: Union[float, float] = [-1.0, 1.0],
                 thetadot_safety_bounds: Union[float, float] = [-np.inf, np.inf],
                 theta_des: float = [0.],
                 torque_bounds: Union[float, float] = [-15., 15.],
                 max_steps: int = 200
                 ):
        super(InvertedPendulum, self).__init__()

        self._tau = tau
        self.g = g
        self.l = l
        self.m = m
        self.torque_bounds = torque_bounds
        self.initial_state = initial_state
        self.theta_safety_bounds = theta_safety_bounds
        self.thetadot_safety_bounds = thetadot_safety_bounds
        self.x1_des = 0.
        self.action_space = spaces.Box(self.torque_bounds[0], self.torque_bounds[1], shape=(1,), dtype=np.float64)
        high = np.array([1.0, 1.0, 8.0])
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        # self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64)
        # self._state = np.array(self.initial_state)
        self.seed()
        self.max_steps = max_steps
        self.count = 0
        self.action_dim=1
        
        # rendering stuff
        self.viewer = None


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
        # return self._state
        return self._get_obs()

    def seed(self, seed=None):
        # not used
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    ### New reward
    def _angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def reward(self, action):
        cost = self._angle_normalize(self._state[0])**2 + 0.1 * self._state[1]**2 \
                + 0.001 * (action ** 2)
        return -float(cost)
    ### end new reward

    def _get_obs(self):
        theta, thetadot = self._state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def step(self, action: float):
        self.count += 1
        action = np.clip(action, self.torque_bounds[0], self.torque_bounds[1]) 
        self.last_u = action # for use in rendering
        c1 = ((3 * self.g)/(2 * self.l))
        c2 = (3 /(self.m * (self.l ** 2)))
        theta, thetadot = self._state[0], self._state[1]
        theta_new = theta + (self._tau * thetadot) + (self._tau ** 2) * ( c1 * np.sin(theta) + c2 * action)
        thetadot_new = thetadot + (self._tau) * ( c1 * np.sin(theta) + c2 * action)
        self._state = np.array([theta_new, thetadot_new], dtype=float)

        theta_min, theta_max = self.theta_safety_bounds[0], self.theta_safety_bounds[1]
        thetadot_min, thetadot_max = self.thetadot_safety_bounds
        done = False

        # if (theta_new < theta_min) or \
        #    (theta_new > theta_max) or \
        #    (thetadot_new < thetadot_min) or \
        #    (thetadot_new > thetadot_max) or \
        #    (self.count > self.max_steps):
        #     done = True

        if self.count > self.max_steps:
            done = True
            
        return self._get_obs().flatten(), self.reward(action), done, {}

        ### OLD
        # return self._state, self.reward(action), done, {}
        
    ### OLD
    # def reset(self):
    #     self._state = np.array(self.initial_state)
    #     self.count = 0
    #     return self._state)

    def reset(self):
        self._state = np.array(self.initial_state)
        # self._state = np.random.uniform(-np.pi, np.pi, size=(2,))
        self.count = 0
        return self._get_obs()

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self._state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    ### OLD
    # def render(self, mode='console'):
    #     if mode != 'console'metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}:
    #         raise NotImplementedError()
    #     print("not implemented")

    # def close(self):
    #    pass
    
    def cbf(self, state=None, eta: float = 0.99):
        """
        Calculates CBF constraint set at a given state. Default is
        the current state.
        """

        state = state if state is not None else self._state

        if (eta>1-1e-3) or (eta<1e-5):
            raise ValueError("eta should be inside (0, 1)")
        c1 = ((3 * self.g)/(2 * self.l))
        c2 = (3 /(self.m * (self.l ** 2)))
        #theta, thetadot = np.arcsin(obs[i][1]), obs[i][2]
        #theta, thetadot = np.arcsin(state[1]), state[2]
        #the above line can replace line 185
        theta, thetadot = state[0], state[1]
        theta_min, theta_max = self.theta_safety_bounds[0], self.theta_safety_bounds[1]
        thetadot_min, thetadot_max = self.thetadot_safety_bounds[0], self.thetadot_safety_bounds[1]
        u_min1 = (1/c2) * (((1 / (self._tau **2)) * (-eta * (theta - theta_min) - self._tau * thetadot)) - c1 * np.sin(theta) )
        u_max1 = (1/c2) * (((1 / (self._tau **2)) * ( eta * (theta_max - theta) - self._tau * thetadot)) - c1 * np.sin(theta) )

        
        u_min2 = (1/c2) * (((1 / (self._tau)) * (-eta * (thetadot - thetadot_min))) - c1 * np.sin(theta) )
        u_max2 = (1/c2) * (((1 / (self._tau)) * ( eta * (thetadot_max - thetadot))) - c1 * np.sin(theta) )

        u_min = max(u_min1, u_min2, self.torque_bounds[0])
        u_max = min(u_max1, u_max2, self.torque_bounds[1])
        
        u_min=self.torque_bounds[0]
        u_max=self.torque_bounds[1]
        if u_min>u_max:
            raise ValueError("Infeasible")
        else:
            return [u_min, u_max]