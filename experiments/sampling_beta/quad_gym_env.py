# -*- coding: utf-8 -*-
"""
Spyder Editor
"""

import numpy as np
import gym
import torch
from gym import spaces
from gym.utils import seeding

from cvxopt import matrix
from cvxopt import solvers
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from numpy import linalg as LA
import cvxpy as cp
import copy
from scipy.optimize import minimize


class QuadDynamics(gym.Env):
    def __init__(self,
                 g: float = -9.81,  # FLU
                 m: float = 0.5,
                 L: float = 0.25,
                 k: float = 3e-6,
                 b: float = 1e-7,
                 I: float = np.diag([5e-3, 5e-3, 10e-3]),
                 kd: float = 0.25,
                 dt: float = 0.05,
                 maxrpm: float = 240,
                 Kp=6,
                 Kd=8,
                 goal=np.array([10.0,10.0]),
                 a_d=1,
                 b_d=1,
                 safety_dist=1.0,
                 robot_radius=0.1,
                 is_crash=False,
                 max_x = 30.0,
                 min_x = -25.0,
                 max_y = 30.0,
                 min_y = -25.0,
                 use_safe=True,
                 max_steps: int = 1000,
                 umax=20*np.array([1, 1]),
                 umin=-20*np.array([1, 1]),
                 env_cbf = False,
                 episodes = 0,
                 layer_size=0,
                 entropy=0,
                 lr=0,
                 device_run = 'server',
                 date = 'enter',
                 run = 0,
                 boundary_tol=1.0
                 ):
        super(QuadDynamics, self).__init__()
        maxthrust = k*np.sum(np.array([maxrpm**2] * 4))
        self.param_dict = {"g": g, "m": m, "L": L, "k": k, "b": b, "I": I,
                           "kd": kd, "dt": dt, "maxRPM": maxrpm, "maxthrust": maxthrust}
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64)
        high = np.array([15, 15, 0, 0.5, 0.5, 0, 2, 2, 0, 0])
        low = np.array([0, 0, 0, -0.5, -0.5, 0, 2, 2, 0, 0])
        self.observation_space = spaces.Box(
            low, high, shape=(10,), dtype=np.float32)
        self.shape_dict = {}
        self.K = np.array([Kp, Kd])
        self.obstacle = np.array([4.5, 4.5])
        self.obstacle_vel = np.array([0, 0])
        self.use_safe = use_safe
        self.goal = goal
        self.m = m
        self.g = g
        self.k = k
        self.kd = kd
        self.I = I
        self.L = L
        self.b = b
        self.dt = dt
        self.a_d = 1
        self.b_d = 1
        self.safety_dist = safety_dist
        self.count = 0
        self.max_steps = max_steps
        self.num_obstacles=1
        self.robot_radius = robot_radius
        #controller limits
        self.umax = umax
        self.umin = umin
        self.x_opt=0 #for optimization purposes
        self.y_opt=0 #for optimization purposes
        self.reward_tol=0.25
        self.env_cbf=env_cbf
        self.episodes = episodes,
        self.entropy=entropy,
        self.lr=lr,
        self.layer_size=layer_size,
        self.device_run = device_run,
        self.date = date,
        self.run = run,
        self.max_x = max_x,
        self.max_y = max_y,
        self.min_x = min_x,
        self.min_y = min_y,
        self.boundary_tol = boundary_tol,
        self.failure_flag=1


    def step_dynamics(self, u):
        """Step dynamics given current state and input. Updates state dict.

        Parameters
        ----------
        state : dict 
            contains current x, xdot, theta, thetadot
        u : (4, ) np.ndarray
            control input - (angular velocity)^squared of motors (rad^2/s^2)
        Updates
        -------
        state : dict 
            updates with next x, xdot, theta, thetadot  
        """
        # Compute angular velocity vector from angular velocities
        omega = self.thetadot2omega(
            self.state["thetadot"], self.state["theta"])

        # linear and angular accelerations given input and state
        a = self.calc_acc(u, self.state["theta"], self.state["xdot"])
        omegadot = self.calc_ang_acc(u, omega)

        # next state
        omega = omega + self.dt * omegadot
        thetadot = self.omega2thetadot(omega, self.state["theta"])
        theta = self.state["theta"] + self.dt * self.state["thetadot"]
        xdot = self.state["xdot"] + self.dt * a
        x = self.state["x"] + self.dt * xdot
        
        #keeping z axis at 0
        if x[2]<=0.0:
            x[2] = 0.0
            xdot[2] = 0.0 
        # Update state dictionary
        self.state["x"] = x
        self.state["xdot"] = xdot
        self.state["theta"] = theta
        self.state["thetadot"] = thetadot

        return self.state

    def compute_thrust(self, u):
        """Compute total thrust (in body frame) given control input and thrust coefficient. Used in calc_acc().
        Clips if above maximum rpm (10000).
        thrust = k * sum(u)

        Parameters
        ----------
        u : (4, ) np.ndarray
            control input - (angular velocity)^squared of motors (rad^2/s^2)
        k : float
            thrust coefficient
        Returns
        -------
        T : (3, ) np.ndarray
            thrust in body frame
        """
        u = np.clip(u, 0, self.param_dict["maxRPM"]**2)
        T = np.array([0, 0, self.k*np.sum(u)])

        return T

    def calc_torque(self, u):
        """Compute torque (body-frame), given control input, and coefficients. Used in calc_ang_acc()

        Parameters
        ----------
        u : (4, ) np.ndarray
            control input - (angular velocity)^squared of motors (rad^2/s^2)
        L : float
            distance from center of quadcopter to any propellers, to find torque (m).

        b : float # TODO: description

        k : float
            thrust coefficient
        Returns
        -------
        tau : (3,) np.ndarray
            torque in body frame (Nm)
        """
        tau = np.array([
            self.L * self.k * (u[0]-u[2]),
            self.L * self.k * (u[1]-u[3]),
            self.b * (u[0]-u[1] + u[2]-u[3])
        ])

        return tau

    def calc_acc(self, u, theta, xdot):
        """Computes linear acceleration (in inertial frame) given control input, gravity, thrust and drag.
        a = g + T_b+Fd/m
        Parameters
        ----------
        u : (4, ) np.ndarray
            control input - (angular velocity)^squared of motors (rad^2/s^2)
        theta : (3, ) np.ndarray 
            rpy angle in body frame (radian) 
        xdot : (3, ) np.ndarray
            linear velocity in body frame (m/s), for drag calc 
        m : float
            mass of quadrotor (kg)
        g : float
            gravitational acceleration (m/s^2)
        k : float
            thrust coefficient
        kd : float
            drag coefficient
        Returns
        -------
        a : (3, ) np.ndarray 
            linear acceleration in inertial frame (m/s^2)
        """
        gravity = np.array([0, 0, self.g])
        R = self.get_rot_matrix(theta)
        thrust = self.compute_thrust(u)
        T = np.dot(R, thrust)
        Fd = -self.kd * xdot
        a = gravity + 1/(self.m) * T + Fd
        return a

    def calc_ang_acc(self, u, omega):
        """Computes angular acceleration (in body frame) given control input, angular velocity vector, inertial matrix.

        omegaddot = inv(I) * (torque - w x (Iw))
        Parameters
        ----------
        u : (4, ) np.ndarray
            control input - (angular velocity)^squared of motors (rad^2/s^2)
        omega : (3, ) np.ndarray 
            angular velcoity vector in body frame
        I : (3, 3) np.ndarray 
            inertia matrix
        L : float
            distance from center of quadcopter to any propellers, to find torque (m).
        b : float # TODO: description
        k : float
            thrust coefficient
        Returns
        -------
        omegaddot : (3, ) np.ndarray
            rotational acceleration in body frame #TODO: units
        """
        # Calculate torque given control input and physical constants
        tau = self.calc_torque(u)

        # Calculate body frame angular acceleration using Euler's equation
        omegaddot = np.dot(np.linalg.inv(
            self.I), (tau - np.cross(omega, np.dot(self.I, omega))))

        return omegaddot

    def omega2thetadot(self, omega, theta):
        """Compute angle rate from angular velocity vector and euler angle.
        Uses Tait Bryan's z-y-x/yaw-pitch-roll.
        Parameters
        ----------
        omega: (3, ) np.ndarray
            angular velocity vector
        theta: (3, ) np.ndarray
            euler angles in body frame (roll, pitch, yaw)
        Returns
        ---------
        thetadot: (3, ) np.ndarray
            time derivative of euler angles (roll rate, pitch rate, yaw rate)
        """
        mult_matrix = np.array(
            [
                [1, 0, -np.sin(theta[1])],
                [0, np.cos(theta[0]), np.cos(theta[1])*np.sin(theta[0])],
                [0, -np.sin(theta[0]), np.cos(theta[1])*np.cos(theta[0])]
            ], dtype='double')

        mult_inv = np.linalg.inv(mult_matrix)
        thetadot = np.dot(mult_inv, omega)

        return thetadot

    def thetadot2omega(self, thetadot, theta):
        """Compute angular velocity vector from euler angle and associated rates.
        ----------
        thetadot: (3, ) np.ndarray
            time derivative of euler angles (roll rate, pitch rate, yaw rate)
        theta: (3, ) np.ndarray
            euler angles in body frame (roll, pitch, yaw)
        Returns
        ---------
        w: (3, ) np.ndarray
            angular velocity vector (in body frame)
        """
        roll = theta[0]
        pitch = theta[1]
        yaw = theta[2]

        mult_matrix = np.array(
            [
                [1, 0, -np.sin(pitch)],
                [0, np.cos(roll), np.cos(pitch)*np.sin(roll)],
                [0, -np.sin(roll), np.cos(pitch)*np.cos(roll)]
            ]
        )
        w = np.dot(mult_matrix, thetadot)
        return w

    def get_rot_matrix(self, angles):
        [phi, theta, psi] = angles
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cthe = np.cos(theta)
        sthe = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)

        rot_mat = np.array([[cthe * cpsi, sphi * sthe * cpsi - cphi * spsi, cphi * sthe * cpsi + sphi * spsi],
                            [cthe * spsi, sphi * sthe * spsi + cphi *
                                cpsi, cphi * sthe * spsi - sphi * cpsi],
                            [-sthe,       cthe * sphi,                      cthe * cphi]])
        return rot_mat

    def reward(self):
        cost = LA.norm(self._get_obs()[:2]-self.goal)
        if cost <= 5*self.reward_tol:
            cost = cost-50.0
        if cost <= self.reward_tol:
            cost = -50.0
        if self._get_obs()[0]<=np.double(self.min_x) + np.double(self.boundary_tol) or self._get_obs()[0]>=np.double(self.max_x)-np.double(self.boundary_tol) or self._get_obs()[1]<=np.double(self.min_y)+np.double(self.boundary_tol) or self._get_obs()[1]>=np.double(self.max_y)-np.double(self.boundary_tol):
            cost = cost+200.0
        return -float(cost)/300

    def step(self, u_hat_acc):
        self.count += 1
        self.update_obstacles()
        u_hat_acc = u_hat_acc.reshape(2,1) 
        u_hat_acc = np.ndarray.flatten(
            np.array(np.vstack((u_hat_acc, np.zeros((1, 1))))))  # acceleration
        assert(u_hat_acc.shape == (3,))
        u_motor = self.go_to_acceleration(u_hat_acc)  # desired motor rate ^2
        # pass this input to the dynamics
        self.state = self.step_dynamics(u_motor)
        done = False

        if self.failure_flag ==1:
            done = True
            self.reset()
            self.failure_flag = 0
        if LA.norm(self._get_obs()[:2]-self.goal) <= self.reward_tol:
            done = True
            self.reset()
        if self.count > self.max_steps:
            done = True
            self.reset()
        if self._get_obs()[0] >= np.double(self.max_x) or self._get_obs()[1] >= np.double(self.max_y) or self._get_obs()[0] <= np.double(self.min_x) or self._get_obs()[1]<=np.double(self.min_y):
            done = True
            self.reset()  

        return self._get_obs(), self.reward(), done, {}

    def render(self, *args, **kwargs):
        raise NotImplementedError()

    def reset(self):
        """Initialize state dictionary. """

        self.state = {"x": np.array([-1.5, -1.5, 0]),
                       "xdot": np.array([0, 0, 0]),
                       "theta": np.radians(np.array([0, 0, 0])),
                       "thetadot": np.radians(np.array([0, 0, 0]))
                       }
        mod_obs = self._get_obs()
        self.count = 0
        return mod_obs

    def _get_obs(self):

        index=0
        index_temp=0
        
        num_states=self.state["x"].size+self.state["xdot"].size+4*(self.num_obstacles)
        aug_obs=np.empty(num_states) #initialization for the augmented observation
        
        for j in range(self.state["x"].size):
            aug_obs[j]=self.state["x"][j]
            index_temp=j
        index=index_temp+1
        
        for j in range(self.state["x"].size):
            aug_obs[j+index]=self.state["xdot"][j]
            index_temp=j
        index=index_temp+index
        
        for i in range(self.num_obstacles):
            aug_obs[index+1] = float(self.obstacle[0])
            aug_obs[index+2] = float(self.obstacle[1])
            aug_obs[index+3] = float(self.obstacle_vel[0])
            aug_obs[index+4] = float(self.obstacle_vel[1])
        return aug_obs

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def compute_h(self, observation, obstacles):  # why pass such an obstacle? --vipul
        h = np.zeros((obstacles.shape[1], 1))
        for i in range(obstacles.shape[1]):
            rel_r = observation[:2].reshape(2,1) - obstacles[:, i].reshape(2, 1)

            hr = self.h_func(rel_r[0], rel_r[1], self.a_d,
                             self.b_d, self.safety_dist)
            h[i] = hr
        return h


    def compute_hd(self, observation, observation_dot, obstacles, obstacles_vel):
        hd = np.zeros((obstacles.shape[1], 1))
        for i in range(obstacles.shape[1]):
            rel_r = observation[:2].reshape(2,1) - obstacles[:, i].reshape(2, 1)
            rd = observation_dot[:2].reshape(2,1) - obstacles_vel[:, i].reshape(2, 1)    
            term1 = (4 * np.power(rel_r[0], 3) * rd[0])/(np.power(self.a_d, 4))
            term2 = (4 * np.power(rel_r[1], 3) * rd[1])/(np.power(self.b_d, 4))
            hd[i] = term1 + term2
        return hd

    def compute_A(self, observation, obstacles):  # no modifications -vipul
        A = np.empty((0, 2))
        for i in range(obstacles.shape[1]):
            rel_r = observation[:2].reshape(2,1) - obstacles[:, i].reshape(2, 1) # issue due to '-' not being compatible here
            A0 = (4 * np.power(rel_r[0], 3))/(np.power(self.a_d, 4))
            A1 = (4 * np.power(rel_r[1], 3))/(np.power(self.b_d, 4))
            Atemp = np.array([np.hstack((A0, A1))])
            A = np.array(np.vstack((A, Atemp)))
        A = -1 * matrix(A.astype(np.double), tc='d')
        return A

    def compute_h_hd(self, observation, observation_dot, obstacles, obstacles_vel):
        h = self.compute_h(observation, obstacles)
        hd = self.compute_hd(observation, observation_dot,
                             obstacles, obstacles_vel)
        return np.vstack((h, hd)).astype(np.double)

    def compute_b(self, observation, observation_dot, obstacles, obstacles_vel):
        """extra + K * [h hd]"""
        rel_r = observation[:2].reshape(2,1) - obstacles
        rd = observation_dot[:2] - obstacles_vel

        extra = -((12 * np.square(rel_r[0]) * np.square(rd[0]))/np.power(self.a_d, 4) + (
            12 * np.square(rel_r[1]) * np.square(rd[1]))/np.power(self.b_d, 4))
        extra = extra.reshape(obstacles.shape[1], 1)

        b_ineq = extra - (self.K[0] * self.compute_h(observation, obstacles) + self.K[1]
                          * self.compute_hd(observation, observation_dot, obstacles, obstacles_vel))
        b_ineq = -b_ineq
        return b_ineq

    def cbf(self, mod_obs):  # cbf for a quadrotor with static obstacle(s)
        # control in R^2
        # obs = obs if obs is not None else self._get_obs #original--vipul
        # mod_obs is modified observation, it consists of x,y,z appended with xdot, ydot, zdot

        assert not torch.is_tensor(mod_obs), 'no tensors allowed'

        observation = mod_obs[:3]
        observation=observation.reshape(observation.shape[0],1)
        observation_dot = mod_obs[3:6]
        observation_dot=observation_dot.reshape(observation_dot.shape[0],1)
        obstacles = mod_obs[6:8]
        obstacles=obstacles.reshape(obstacles.shape[0],1)
        obstacles_vel = mod_obs[8:10]
        obstacles_vel=obstacles_vel.reshape(obstacles_vel.shape[0],1)
        A = self.compute_A(observation, obstacles)
        b = self.compute_b(observation, observation_dot,
                           obstacles, obstacles_vel)
        # Starting for umin and umax
        umin=np.empty((2,1))
        umax=np.empty((2,1))
        flag_xmin_ymin=0
        flag_xmin_ymax=0
        flag_xmax_ymin=0
        flag_xmax_ymax=0
        failure_flag=0
        def half_plane_interval_intersection(a_1, a_2, b, x_min, x_max, y_min, y_max, flag_xmin_ymax, flag_xmin_ymin, flag_xmax_ymin, flag_xmax_ymax):
            """
            Find the intersection between a half-plane defined by a_1x + a_2y <= b
            and a 2D interval [x_min, x_max] x [y_min, y_max].
            """
            intersections = []
            umin_temp=np.zeros((2,1))
            umax_temp=np.zeros((2,1))
            # Check if the half-plane intersects the left or right edge of the interval
            if a_1 != 0:
                x_left = (b - a_2 * y_min) / a_1
                x_right = (b - a_2 * y_max) / a_1
                if x_min <= x_left <= x_max:
                    intersections.append((np.double(x_left), np.double(y_min)))
                    if a_1*(np.double(x_left) - 0.00000001)+a_2*np.double(y_min)<=b:
                        umin_temp[0]=x_min
                        umin_temp[1]=y_min
                        flag_xmin_ymin=1
                    if a_1*(np.double(x_left) + 0.00000001)+a_2*np.double(y_min)<=b:
                        umax_temp[0]=x_max
                        umin_temp[1]=y_min
                        flag_xmax_ymin=1
                if x_min <= x_right <= x_max:
                    intersections.append((np.double(x_right), np.double(y_max)))
                    if a_1*(x_right - 0.00000001)+a_2*y_max<=b:
                        umin_temp[0]=x_min
                        umax_temp[1]=y_max
                        flag_xmin_ymax=1
                    if a_1*(x_right + 0.00000001)+a_2*y_max<=b:
                        umax_temp[0]=x_max
                        umax_temp[1]=y_max
                        flag_xmax_ymax=1
            # Check if the half-plane intersects the top or bottom edge of the interval
            if a_2 != 0:
                y_bottom = (b - a_1 * x_min) / a_2
                y_top = (b - a_1 * x_max) / a_2
                if y_min <= y_bottom <= y_max:
                    intersections.append((np.double(x_min), np.double(y_bottom)))
                    if a_1*x_min+a_2*(y_bottom - 0.00000001)<=b:
                        umin_temp[0]=x_min
                        umin_temp[1]=y_min
                        flag_xmin_ymin=1
                    if a_1*x_min+a_2*(y_bottom + 0.00000001)<=b:
                        umin_temp[0]=x_min
                        umax_temp[1]=y_max   
                        flag_xmin_ymax=1
                if y_min <= y_top <= y_max:
                    intersections.append((np.double(x_max), np.double(y_top)))
                    if a_1*x_max+a_2*(y_top - 0.00000001)<=b:
                        umax_temp[0]=x_max
                        umin_temp[1]=y_min
                        flag_xmax_ymin=1
                    if a_1*x_max+a_2*(y_top + 0.00000001)<=b:
                        umax_temp[0]=x_max
                        umax_temp[1]=y_max   
                        flag_xmax_ymax=1
            intersections_temp=np.array(intersections)
            if intersections_temp.shape==(2,2):
                intersections=np.array(intersections)
            return intersections, flag_xmin_ymax, flag_xmin_ymin, flag_xmax_ymin, flag_xmax_ymax
        intersections, flag_xmin_ymax, flag_xmin_ymin, flag_xmax_ymin, flag_xmax_ymax = half_plane_interval_intersection(A[0], A[1], b, self.umin[0], self.umax[0], self.umin[1], self.umax[1], flag_xmin_ymax, flag_xmin_ymin, flag_xmax_ymin, flag_xmax_ymax)
        intersections_temp=np.array(intersections)
        if intersections_temp.shape!=(2,2):
            intersections.append((self.umin[0],self.umin[1]))
            intersections.append((self.umax[0],self.umax[1]))

        #Setting up the optimization formulation
        
        x = cp.Variable(pos=True, name="x")
        y = cp.Variable(pos=True, name="y")
        flag_max_min=0
        flag_max_max=0
        flag_min_min=0
        flag_min_max=0
        flag_none=1
        if flag_xmin_ymax==1 and flag_xmin_ymin==0 and flag_xmax_ymin==0 and flag_xmax_ymax==0:
            #objective=(x-self.umin[0])*(self.umax[1]-y)
            flag_min_max=1
            flag_none=0
        elif flag_xmin_ymax==1 and flag_xmin_ymin==1 and flag_xmax_ymin==0 and flag_xmax_ymax==0: #case 2, need slope
            flag_min_min=1
            flag_none=0
            
        #considering slope for case 2
        elif flag_xmin_ymax==1 and flag_xmin_ymin==1 and flag_xmax_ymin==0 and flag_xmax_ymax==0 and A[0]*A[1]<0: #case 2, using slope
            flag_min_max=1
            flag_none=0
        #done case 2
            
        elif flag_xmin_ymax==1 and flag_xmax_ymin==1 and flag_xmin_ymin==0 and flag_xmax_ymax==0:
            flag_min_min=1
            flag_none=0
        elif flag_xmin_ymin==1 and flag_xmax_ymin==1 and flag_xmax_ymax==0 and flag_xmin_ymax==0: #case 4, need slope
            flag_min_min=1
            flag_none=0
            
        #considering slope for case 4
        elif flag_xmin_ymin==1 and flag_xmax_ymin==1 and flag_xmax_ymax==0 and flag_xmin_ymax==0 and A[0]*A[1]<0: #case 4, using slope
            flag_max_min=1
            flag_none=0
        #done case 4
            
        elif flag_xmin_ymin==1 and flag_xmin_ymax==0 and flag_xmax_ymin==0 and flag_xmax_ymax==0:
            flag_min_min=1
            flag_none=0
        elif flag_xmax_ymin==1 and flag_xmax_ymax==0 and flag_xmin_ymin==0 and flag_xmin_ymax==0:
            flag_min_min=1
            flag_none=0
        elif flag_xmax_ymax==1 and flag_xmin_ymin==1 and flag_xmax_ymin==0 and flag_xmin_ymax==0:
            flag_max_min=1
            flag_none=0
        elif flag_xmin_ymax==1 and flag_xmax_ymin==1 and flag_xmin_ymin==0 and flag_xmax_ymax==0: #case 8, need slope
            flag_max_max=1
            flag_none=0
        
        #considering slope for case 8
        elif flag_xmin_ymax==1 and flag_xmax_ymin==1 and flag_xmin_ymin==0 and flag_xmax_ymax==0 and A[0]*A[1]<0: #case 8, using slope
            flag_max_min=1
            flag_none=0
        #done case 8            
            
        elif flag_xmax_ymax==1 and flag_xmax_ymin==0 and flag_xmin_ymin==0 and flag_xmin_ymax==0:
            flag_max_max=1
            flag_none=0
        elif flag_xmin_ymax==1 and flag_xmax_ymax==1 and flag_xmin_ymin==0 and flag_xmax_ymin==0: #case 10, need slope
            flag_max_max=1
            flag_none=0
        
        #considering slope for case 10
        elif flag_xmin_ymax==1 and flag_xmax_ymax==1 and flag_xmin_ymin==0 and flag_xmax_ymin==0 and A[0]*A[1]<0: #case 10, using slope
            flag_min_max=1
            flag_none=0
        #done case 10    
            
        elif flag_xmin_ymax==1 and flag_xmax_ymin==1 and flag_xmin_ymin==0 and flag_xmax_ymax==0:
            flag_max_max=1
            flag_none=0
        elif flag_xmax_ymax==1 and flag_xmin_ymin==1 and flag_xmax_ymin==0 and flag_xmin_ymax==0:
            flag_min_max=1
            flag_none=0
        
        umin_x=copy.deepcopy(self.umin[0])
        umin_y=copy.deepcopy(self.umin[1])
        umax_y=copy.deepcopy(self.umax[1])
        umax_x=copy.deepcopy(self.umax[0])
        if flag_none==0:    
            objective=x*y
            epsilon=0.000000001
            if flag_min_min==1:
                c=np.double(b-A[0]*self.umin[0]-A[1]*self.umin[1])
                constraints=[
                    ((A[0])*x+(A[1])*y)<=c, epsilon<=x, x<=self.umax[0]-self.umin[0], epsilon<=y, y<=self.umax[1]-self.umin[1]] 
                def constraint(x):
                    a1 = A[0]  # coefficient of x
                    a2 = A[1]  # coefficient of y
                    return c - a1 * x[0] - a2 * x[1]
            elif flag_min_max==1:
                c=np.double(b-A[0]*self.umin[0]-A[1]*self.umax[1])
                constraints=[
                    ((A[0])*x+(-A[1])*y)<=c, epsilon<=x, x<=self.umax[0]-self.umin[0], epsilon<=y, y<=self.umax[1]-self.umin[1]]
                def constraint(x):
                    a1 = A[0]  # coefficient of x
                    a2 = -A[1]  # coefficient of y
                    return c - a1 * x[0] - a2 * x[1]
            elif flag_max_max==1:
                c=np.double(b-A[0]*self.umax[0]-A[1]*self.umax[1])
                constraints=[
                    ((-A[0])*x+(-A[1])*y)<=c, epsilon<=x, x<=self.umax[0]-self.umin[0], epsilon<=y, y<=self.umax[1]-self.umin[1]]
                def constraint(x):
                    a1 = -A[0]  # coefficient of x
                    a2 = -A[1]  # coefficient of y
                    return c - a1 * x[0] - a2 * x[1]
            elif flag_max_min==1:
                c=np.double(b-A[0]*self.umax[0]-A[1]*self.umin[1])
                constraints=[
                    ((-A[0])*x+(A[1])*y)<=c, epsilon<=x, x<=self.umax[0]-self.umin[0], epsilon<=y, y<=self.umax[1]-self.umin[1]]
                def constraint(x):
                    a1 = -A[0]  # coefficient of x
                    a2 = A[1]  # coefficient of y
                    return c - a1 * x[0] - a2 * x[1]             
            try:     
                assert objective.is_log_log_concave()
                assert all(constraint.is_dgp() for constraint in constraints)
                problem = cp.Problem(cp.Maximize(objective), constraints)
                problem.solve(gp=True, verbose=False)    
                x_opt=x.value
                y_opt=y.value
                
            #except AssertionError as msg:
            except:   
                try:
                    def objective(x):
                        return -x[0] * x[1]
    
                    # Define the initial guess
                    x0 = np.array([self.x_opt, self.y_opt])

                    bounds=[(0, self.umax[0]-self.umin[0]), (0, self.umax[1]-self.umin[1])]
                    # Solve the optimization problem
                    
                    methods_list = ['trust-constr','SLSQP','Nelder-Mead','Powell','CG','BFGS','Newton-CG',
                                   'L-BFGS-B','TNC','COBYLA','dogleg','trust-ncg',
                                   'trust-exact','trust-krylov']
                    for methods in methods_list:
                        result = minimize(objective, x0, method=methods, constraints={'type': 'ineq', 'fun': constraint}, bounds=bounds)
                        if result.success:
                            # Retrieve the optimal values
                            x_opt = result.x[0]
                            y_opt = result.x[1]
                            break
                        else:
                            #import pdb; pdb.set_trace()
                            failure_flag=1
                            print("Optimization failed for all the solvers. Message:", result.message)
                            print(A)
                            print(b)
                            print(observation)
                            print(observation_dot)
                            x_opt=self.x_opt
                            y_opt=self.y_opt
                except:
                    failure_flag=1
                    print(A)
                    print(b)
                    print(observation)
                    print(observation_dot)
                    print("Second Exception Failed!! Optimization failed for everything. Message:", result.message)
                    x_opt=self.x_opt
                    y_opt=self.y_opt
                    
            self.x_opt=copy.deepcopy(x_opt)
            self.y_opt=copy.deepcopy(y_opt)
            
            if flag_min_min==1:
                x_true=x_opt+self.umin[0]
                y_true=y_opt+self.umin[1]
                umin[0]=umin_x
                umin[1]=umin_y
                umax[0]=x_true
                umax[1]=y_true
            elif flag_min_max==1:
                x_true=x_opt+self.umin[0]
                y_true=self.umax[1]-y_opt
                umin[0]=umin_x
                umin[1]=y_true
                umax[0]=x_true
                umax[1]=umax_y
            elif flag_max_max==1:
                x_true=self.umax[0]-x_opt
                y_true=self.umax[1]-y_opt
                umin[0]=x_true
                umin[1]=y_true
                umax[0]=umax_x
                umax[1]=umax_y
            elif flag_max_min==1:
                x_true=self.umax[0]-x_opt
                y_true=y_opt+self.umin[1]
                umin[0]=x_true
                umin[1]=umin_y
                umax[0]=umax_x
                umax[1]=y_true
        if flag_none==1:
            umin[0]=umin_x
            umin[1]=umin_y
            umax[0]=umax_x
            umax[1]=umax_y
        if failure_flag == 1 and (self._get_obs()[0] - self.obstacle[0])**4/(self.a_d)**4 + (self._get_obs()[1] - self.obstacle[1])**4/(self.b_d)**4 < 1.2*self.safety_dist:
            self.failure_flag=1
            umin=0.0*self.umin
            umax=0.0*self.umax
        umax=umax.flatten()
        umin=umin.flatten()
        return [umin, umax]

    def compute_nom_control(self, Kn=np.array([-0.08, -0.2])):
        vd = Kn[0]*(np.atleast_2d(self.state["x"][:2]).T - self.goal)
        u_nom = Kn[1]*(np.atleast_2d(self.state["xdot"][:2]).T - vd)

        if np.linalg.norm(u_nom) > 0.05:
            u_nom = (u_nom/np.linalg.norm(u_nom)) * 0.05
        return matrix(u_nom, tc='d')


    def update_obstacles(self, noisy=False):
        obstacle = []
        obstacle_vel = []
        if not len(self.obstacle):
            return self.obstacles, self.obstacles_vel

        if self.num_obstacles == 1: 
            obstacle.append(self.obstacle.reshape(2, 1))
            self.obstacle = np.asarray(obstacle).reshape(2, 1)
            obstacle_vel.append(np.array([0, 0]))
            self.obstacle_vel = np.asarray(obstacle_vel).reshape(2, 1)
            return self.obstacle, self.obstacle_vel
        for i in range(self.obstacle.shape[0]):
            self.obstacle.append(self.obstacle[i].reshape(2, 1))
            self.obstacle_vel.append(np.array([0, 0]))

        return self.obstacle, self.obstacle_vel

    @np.vectorize
    def h_func(r1, r2, a, b, safety_dist):
        hr = np.power(r1, 4)/np.power(a, 4) + \
            np.power(r2, 4)/np.power(b, 4) - safety_dist
        return hr



    def go_to_acceleration(self, des_acc):
        # pass
        des_theta, des_thrust_pc = self.dynamic_inversion(des_acc)
        u = self.pi_attitude_control(
            des_theta, des_thrust_pc)  # attitude control
        return u

    def dynamic_inversion(self, des_acc):
        """Invert dynamics. For outer loop, given v_tot, compute attitude.
        Similar to control allocator.
        TODO: do 1-1 mapping?
        Parameters
        ----------
        self.v_tot
            total v: v_cr + v_lc - v_ad
        state #TODO: use self.x
            state
        Returns
        -------
        desired_theta: np.ndarray(3,)
            desired roll, pitch, yaw angle (rad) to attitude controller
        """
        yaw = self.state["theta"][2]
        U1 = np.linalg.norm(des_acc - np.array([0, 0, self.param_dict["g"]]))
        des_pitch_noyaw = np.arcsin(des_acc[0] / U1)
        des_angle = [des_pitch_noyaw,
                     np.arcsin(des_acc[1] / (U1 * np.cos(des_pitch_noyaw)))]
        des_pitch = des_angle[0] * np.cos(yaw) + des_angle[1] * np.sin(yaw)
        des_roll = des_angle[0] * np.sin(yaw) - des_angle[1] * np.cos(yaw)

        des_pitch = np.clip(des_pitch, np.radians(-30), np.radians(30))
        des_roll = np.clip(des_roll, np.radians(-30), np.radians(30))
        des_yaw = yaw
        des_theta = [des_roll, des_pitch, des_yaw]

        thrust = (self.param_dict["m"] * (des_acc[2] -
                                          self.param_dict["g"]))/self.param_dict["k"]  # T=ma/k
        max_tot_u = 400000000.0
        des_thrust_pc = thrust/max_tot_u

        return des_theta, des_thrust_pc

    def go_to_position(self, des_pos, integral_p_err=None, integral_v_err=None):  # modified--vipul

        des_vel, integral_p_err = self.pi_position_control(
            des_pos, integral_p_err)
        des_thrust, des_theta, integral_v_err = self.pi_velocity_control(
            des_vel, integral_v_err)  # attitude control
        u = self.pi_attitude_control(des_theta, des_thrust)  # attitude control

        return u

    def pi_position_control(self, des_pos, integral_p_err=None):  # modified --vipul
        if integral_p_err is None:
            integral_p_err = np.zeros((3,))

        Px = -0.5
        Ix = 0  # -0.005
        Py = -0.5
        Iy = 0  # 0.005
        Pz = -1

        [x, y, z] = self.state["x"]
        [x_d, y_d, z_d] = des_pos
        yaw = self.state["theta"][2]

        # Compute error
        p_err = self.state["x"] - des_pos
        # accumulate error integral
        integral_p_err += p_err

        # Get PID Error
        # TODO: vectorize

        pid_err_x = Px * p_err[0] + Ix * integral_p_err[0]
        pid_err_y = Py * p_err[1] + Iy * integral_p_err[1]
        pid_err_z = Pz * p_err[2]  # TODO: project onto attitude angle?

        # TODO: implement for z vel
        des_xv = pid_err_x  # * np.cos(yaw) + pid_err_y * np.sin(yaw)
        des_yv = pid_err_y  # * np.sin(yaw) - pid_err_y * np.cos(yaw)

        # TODO: currently, set z as constant
        des_zv = pid_err_z

        return np.array([des_xv, des_yv, des_zv]), integral_p_err

    def pi_velocity_control(self, des_vel, integral_v_err=None):
        """
        Assume desire zero angular velocity? Also clips min and max roll, pitch.

        Parameter
        ---------
        state : dict 
            contains current x, xdot, theta, thetadot

        des_vel : (3, ) np.ndarray
            desired linear velocity

        integral_v_err : (3, ) np.ndarray
            keeps track of integral error

        Returns
        -------
        uv : (3, ) np.ndarray
            roll, pitch, yaw 
        """
        if integral_v_err is None:
            integral_v_err = np.zeros((3,))

        Pxd = -0.12
        Ixd = -0.005  # -0.005
        Pyd = -0.12
        Iyd = -0.005  # 0.005
        Pzd = -0.001
        # TODO: change to return roll pitch yawrate thrust

        [xv, yv, zv] = self.state["xdot"]
        [xv_d, yv_d, zv_d] = des_vel
        yaw = self.state["theta"][2]

        # Compute error
        v_err = self.state["xdot"] - des_vel
        # accumulate error integral
        integral_v_err += v_err

        # Get PID Error
        # TODO: vectorize

        pid_err_x = Pxd * v_err[0] + Ixd * integral_v_err[0]
        pid_err_y = Pyd * v_err[1] + Iyd * integral_v_err[1]
        pid_err_z = Pzd * v_err[2]  # TODO: project onto attitude angle?

        tot_u_constant = 408750 * 4  # hover, for four motors
        max_tot_u = 400000000.0
        thrust_pc_constant = tot_u_constant/max_tot_u
        des_thrust_pc = thrust_pc_constant + pid_err_z

        des_pitch = pid_err_x * np.cos(yaw) + pid_err_y * np.sin(yaw)
        des_roll = pid_err_x * np.sin(yaw) - pid_err_y * np.cos(yaw)

        # TODO: move to attitude controller?
        des_pitch = np.clip(des_pitch, np.radians(-30), np.radians(30))
        des_roll = np.clip(des_roll, np.radians(-30), np.radians(30))

        # TODO: currently, set yaw as constant
        des_yaw = self.state["theta"][2]

        return des_thrust_pc, np.array([des_roll, des_pitch, self.state["theta"][2]]), integral_v_err

    def pi_attitude_control(self, des_theta, des_thrust_pc):
        """Attitude controller (PD). Uses current theta and theta dot.

        Parameter
        ---------
        state : dict 
            contains current x, xdot, theta, thetadot

        k : float
            thrust coefficient

        Returns
        -------
        u : (4, ) np.ndarray
            control input - (angular velocity)^squared of motors (rad^2/s^2)

        """

        Kd = 10  # why define these parameters again and that too with different values
        Kp = 30

        theta = self.state["theta"]
        thetadot = self.state["thetadot"]

        max_tot_u = 400000000.0
        tot_u = des_thrust_pc * max_tot_u

        e = Kd * thetadot + Kp * (theta - des_theta)

        u = self.angerr2u(e, theta, tot_u)
        return u


    def angerr2u(self, error, theta, tot_thrust):
        """Compute control input given angular error. Closed form specification
        with dynamics inversion.

        Parameters
        ----------
        error
        """
        L = self.param_dict["L"]
        k = self.param_dict["k"]
        b = self.param_dict["b"]
        I = self.param_dict["I"]

        e0 = error[0]
        e1 = error[1]
        e2 = error[2]
        Ixx = I[0, 0]
        Iyy = I[1, 1]
        Izz = I[2, 2]

        # TODO: make more readable
        r0 = tot_thrust/4 - (2*b*e0*Ixx + e2*Izz*k*L)/(4*b*k*L)

        r1 = tot_thrust/4 + (e2*Izz)/(4*b) - (e1*Iyy)/(2*k*L)

        r2 = tot_thrust/4 + (2*b*e0*Ixx - e2*Izz*k*L)/(4*b*k*L)

        r3 = tot_thrust/4 + (e2*Izz)/(4*b) + (e1*Iyy)/(2*k*L)

        return np.array([r0, r1, r2, r3])