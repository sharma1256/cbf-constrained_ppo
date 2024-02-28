# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 20:33:42 2023

@author: User
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
# copied from Wes' pendulum example'
from typing import Union
import numpy as np
import gym
import torch
from gym import spaces
from gym.utils import seeding
from os import path
#from gym.envs.classic_control import rendering
import cvxopt
from cvxopt import matrix
from cvxopt import solvers
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time
import warnings
from numpy import linalg as LA
import cvxpy as cp
import copy
from scipy.optimize import minimize
import csv

class QuadDynamicsProj(gym.Env):
    # major changes in __init__
    # because every other class(than quaddynamics) is being dissolved
    # and we need to export their initialization paramters over to the main class i.e. quaddynamics
    def __init__(self,
                 g: float = -9.81,  # FLU
                 m: float = 0.5,
                 L: float = 0.25,
                 k: float = 3e-6,
                 b: float = 1e-7,
                 I: float = np.diag([5e-3, 5e-3, 10e-3]),
                 kd: float = 0.25,
                 dt: float = 0.05,
                 maxrpm: float = 140,
                 Kp=6,
                 Kd=8,
                 goal=np.array([8.0,8.0]),
                 a_d=1,
                 b_d=1,
                 safety_dist=1.0,
                 robot_radius=0.1,
                 is_crash=False,  # Sets title as Crashed when crashed once
                 #this initial state s never used!!!
                 #look at reset function to see the initial state
                 initial_state={"x": np.array([0.0, 0.0, 0]),
                                "xdot": np.zeros(3,),
                                # ! hardcoded
                                "theta": np.radians(np.array([0, 0, 0])),
                                # ! hardcoded
                                "thetadot": np.radians(np.array([0, 0, 0]))
                                },
                 #obstacle=np.array([2, 2]),
                 max_x = 20.0,
                 min_x = -15.0,
                 max_y = 20.0,
                 min_y = -10.0,
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
        # initial state
        self.initial_state = initial_state
        # action and observation space from pendulum code
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64)
        #high = 15.0 * np.ones(shape=(10,)) --original
        high = np.array([15, 15, 0, 0.5, 0.5, 0, 2, 2, 0, 0])
        low = np.array([0, 0, 0, -0.5, -0.5, 0, 2, 2, 0, 0])
        self.observation_space = spaces.Box(
            low, high, shape=(10,), dtype=np.float32)
        #self.initial_state = initial_state
        self.shape_dict = {}  # TODO: a, b
        # I have no idea what this above line means!! -VIPUL
        self.K = np.array([Kp, Kd])
        self.obstacle = np.array([3.8, 3.8])
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

        # Compute linear and angular accelerations given input and state
        # TODO: combine state
        # TODO: move param to member variable
        a = self.calc_acc(u, self.state["theta"], self.state["xdot"])
        omegadot = self.calc_ang_acc(u, omega)

        # Compute next state
        omega = omega + self.dt * omegadot
        thetadot = self.omega2thetadot(omega, self.state["theta"])
        theta = self.state["theta"] + self.dt * self.state["thetadot"]
        xdot = self.state["xdot"] + self.dt * a
        x = self.state["x"] + self.dt * xdot
         #added 07.09.2023 -vipul
         #to handle z going negative!!!
        if x[2]<=0.0:
            x[2] = 0.0
            #xdot[2] = abs(xdot[2])
            xdot[2] = 0.0 #10.02.23 -vipul
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
        #above expression gives us a hint at what the umax and umin should be --vipul
        T = np.array([0, 0, self.k*np.sum(u)])
        # print("u", u)
        # print("T", T)

        return T

    # function definition changed --vipul
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

    # function definition changed --VIPUL

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
        #a = gravity + 1/(self.m) * T + Fd
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

        Uses Tait Bryan's z-y-x/yaw-pitch-roll. 
        Parameters
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

    def reward(self):  # new function --vipul
        cost = LA.norm(self._get_obs()[:2]-self.goal)
        # if cost<=10*self.reward_tol: #get rid of this and change reward_tol = 0.1
        #     cost = 0
        # if self._get_obs()[0] >=3.5:
        #     cost = 0
        if cost <= 5*self.reward_tol:
            cost = 0.00
        if cost <= self.reward_tol:
            cost = -80
        if self._get_obs()[0]<=np.double(self.min_x) + np.double(self.boundary_tol) or self._get_obs()[0]>=np.double(self.max_x)-np.double(self.boundary_tol) or self._get_obs()[1]<=np.double(self.min_y)+np.double(self.boundary_tol) or self._get_obs()[1]>=np.double(self.max_y)-np.double(self.boundary_tol):
            cost = 500
        return -float(cost)/100

    def step(self, u_hat_acc):  # new function --vipul
        self.count += 1
        self.update_obstacles()
        # modify the control input to a suitable data type
        # This reshape is in effect as tensor coming in is a trnaspose of what we actully pass
        #in the dynamics
        u_hat_acc = u_hat_acc.reshape(2,1) 
        
        
        #Projection
        umin, umax = self.cbf_proj(self._get_obs())
        
        # Define the objective function
        def objective_function(u):
            return abs(u[0] - u_hat_acc[0]) + abs(u[1] - u_hat_acc[1])
        
        # Initial guess for u
        initial_guess = [0.0, 0.0]
        
        # Target value for u (2D vector)
        u_s = [u_hat_acc[0], u_hat_acc[1]]
        
        # Define the bounds for the square (S)
        x_bounds = (umin[0], umax[0])  # For the x-coordinate
        y_bounds = (umin[1], umax[1])  # For the y-coordinate
        
        # Define the optimization problem
        optimization_result = minimize(objective_function, initial_guess, bounds=[x_bounds, y_bounds])
        
        # Extract the optimal solution
        u_safe = optimization_result.x
        
        #print("Optimal u:", u_safe)
        u_safe = u_safe.reshape(2,1)
        u_safe = np.ndarray.flatten(
            np.array(np.vstack((u_safe, np.zeros((1, 1))))))  # acceleration
        assert(u_safe.shape == (3,))
        u_motor = self.go_to_acceleration(u_safe)  # desired motor rate ^2
        # time to pass this input to the dynamics
        self.state = self.step_dynamics(u_motor)
        done = False
        # self.state_hist.append(self.state["x"]) --temporarily commenting
        #init_state = deepcopy(self.initial_state["x"][:2])
        #below lines are temporarily commented out
        if self.failure_flag ==1:
            done = True
            #import pdb; pdb.set_trace()
            self.reset()
            self.failure_flag = 0
        if LA.norm(self._get_obs()[:2]-self.goal) <= self.reward_tol:
            done = True
            #import pdb; pdb.set_trace()
            self.reset()
        if self.count > self.max_steps:
            done = True
            #import pdb; pdb.set_trace()
            self.reset()
        # if LA.norm(self._get_obs()[:2]-init_state)>=16
        #     done = True
        #     self.reset()  
        if self._get_obs()[0] >= np.double(self.max_x) or self._get_obs()[1] >= np.double(self.max_y) or self._get_obs()[0] <= np.double(self.min_x) or self._get_obs()[1]<=np.double(self.min_y):
            done = True
            #import pdb; pdb.set_trace()
            self.reset()

        # do we need the flatten with _get_obs? --vipul
        return self._get_obs(), self.reward(), done, {}

    def render(self, *args, **kwargs):
        raise NotImplementedError()
        # new function --vipul

    #     #I don't think this is correct
    #     #but this should atleast plot something
    #     ax1=plt.subplots()
    #     obstacles = []
    #     p = []
    #     x = 0
    #     y = 0
    #     z = 0
    #     sz=0
    #     obstacles=self.update_obstacles(self.obs, noisy=False)
    #     u_hat_acc=self.compute_safe_control(obstacles["obs"],obstacles["obs_v"])
    #     self.plot_step(np.array(obstacles["obs"])[:, :, 0].T, u_hat_acc, ax1) #what is ax1
    #     p.append( self.compute_plot_z(np.array(obstacles["obs"])[:, :, 0].T) )
    #     x = x + p["x"]
    #     y = y + p["y"]
    #     z = z + p["z"]

    #     sz = sz + 1

    #     self.plot_h(x/sz, y/sz, z/sz)
    #     self.plt.pause(0.00000001)

    def reset(self):
        """Initialize state dictionary. """
        #self.state = self.initial_state
        #from copy import deepcopy ,
        #self.state = deepcopy(self.initial_state)
        self.state = {"x": np.array([0.0, 0.0, 0]), #x=4.8 and y=3.5 is a critical point till which the projection based controller works
                       "xdot": np.array([0, 0, 0]),
                       # ! hardcoded
                       "theta": np.radians(np.array([0, 0, 0])),
                       # ! hardcoded
                       "thetadot": np.radians(np.array([0, 0, 0]))
                       }
        mod_obs = self._get_obs()
        self.count = 0
        return mod_obs

    def _get_obs(self):

        ### TODO: get the following to work
        # aug_obs = np.concatenate([
        #     self.state["x"],
        #     self.state["xdot"],
        #     self.obstacle,
        #     self.obstacle_vel,
        # ])

        index=0
        index_temp=0
        
        num_states=self.state["x"].size+self.state["xdot"].size+4*(self.num_obstacles)
        # aug_obs=np.empty([num_states,1]) #initialization for the augmented observation
        aug_obs=np.empty(num_states) #initialization for the augmented observation
        
        for j in range(self.state["x"].size):
            aug_obs[j]=self.state["x"][j]
            index_temp=j
            # x_dim = self.state["x"][0]
            # y_dim = self.state["x"][1]
            # z_dim = self.state["x"][2]
        index=index_temp+1
        
        for j in range(self.state["x"].size):
            aug_obs[j+index]=self.state["xdot"][j]
            index_temp=j
            # x_dim_dot = self.state["xdot"][0]
            # y_dim_dot = self.state["xdot"][1]
            # z_dim_dot = self.state["xdot"][2]
        index=index_temp+index
        
        for i in range(self.num_obstacles):
            aug_obs[index+1] = float(self.obstacle[0])
            aug_obs[index+2] = float(self.obstacle[1])
            aug_obs[index+3] = float(self.obstacle_vel[0])
            aug_obs[index+4] = float(self.obstacle_vel[1])
            # obstacles_x = float(self.obstacle[0])
            # obstacles_y = float(self.obstacle[1])
            # obstacles_velocity_x = float(self.obstacle_vel[0])
            # obstacles_velocity_y = float(self.obstacle_vel[1])

        return aug_obs

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        # not used
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # copying things from ecbf_control.py

    def compute_plot_z(self):  # no modification needed --vipul
        plot_x = np.arange(-7.5, 7.5, 0.4)
        plot_y = np.arange(-7.5, 7.5, 0.4)
        xx, yy = np.meshgrid(plot_x, plot_y, sparse=True)
        z = np.zeros(xx.shape)
        for i in range(self.obstacle.shape[1]):
            ztemp = self.h_func(
                xx - self.obstacle[0][i], yy - self.obstacle[1][i], self.a_d, self.b_d, self.safety_dist) > 0
            z = z + ztemp
        z = z / (self.obstacle.shape[1]-1)
        p = {"x": plot_x, "y": plot_y, "z": z}
        return p

    def plot_h(self, plot_x, plot_y, z):  # no modification needed --vipul
        h = plt.contourf(plot_x, plot_y, z,
                         [-1, 0, 1], colors=['#808080', '#A0A0A0', '#C0C0C0'])
        self.plt.xlabel("X")
        self.plt.ylabel("Y")
        self.plt.pause(0.0001)

    # def compute_h(self, obs=np.array([[0], [0]]).T): # why pass such an obstacle? --vipul

    def compute_h(self, observation, obstacles):  # why pass such an obstacle? --vipul
        h = np.zeros((obstacles.shape[1], 1))
        for i in range(obstacles.shape[1]):
            rel_r = observation[:2].reshape(2,1) - obstacles[:, i].reshape(2, 1)
                #self.state["x"][:2]).T - obstacles[:, i].reshape(2, 1)
                
            # TODO: a, safety_dist, obs, b
            hr = self.h_func(rel_r[0], rel_r[1], self.a_d,
                             self.b_d, self.safety_dist)
            h[i] = hr
        return h

    # no modification needed -- vipul
    def compute_hd(self, observation, observation_dot, obstacles, obstacles_vel):
        hd = np.zeros((obstacles.shape[1], 1))
        for i in range(obstacles.shape[1]):
            rel_r = observation[:2].reshape(2,1) - obstacles[:, i].reshape(2, 1)
                #self.state["x"][:2]).T - obstacles[:, i].reshape(2, 1)
                
            rd = observation_dot[:2].reshape(2,1) - obstacles_vel[:, i].reshape(2, 1)
                #self.state["xdot"][:2]).T - obstacles_vel[:, i].reshape(2, 1)
                
            term1 = (4 * np.power(rel_r[0], 3) * rd[0])/(np.power(self.a_d, 4))
            term2 = (4 * np.power(rel_r[1], 3) * rd[1])/(np.power(self.b_d, 4))
            hd[i] = term1 + term2
        return hd

    def compute_A(self, observation, obstacles):  # no modifications -vipul
        A = np.empty((0, 2))
        for i in range(obstacles.shape[1]):
            #rel_r = np.atleast_2d((self.state["x"][:2]).T - obstacles[:, i].reshape(2, 1)) #-modified due to error while training!
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
        #b_ineq = -1 * matrix(b_ineq.astype(np.double), tc='d')
        #b_ineq = -1 * matrix(b_ineq.astype(np.double), tc='d')
        b_ineq = -b_ineq
        return b_ineq

    def cbf(self, mod_obs):  # cbf for a quadrotor with static obstacle(s)
        # control in R^2
        # obs = obs if obs is not None else self._get_obs #original--vipul
        # mod_obs is modified observation, it consists of x,y,z appended with xdot, ydot, zdot

        assert not torch.is_tensor(mod_obs), 'no tensors allowed'

        # observation = mod_obs[:3]
        # observation=observation.reshape(observation.shape[0],1)
        # observation_dot = mod_obs[3:6]
        # observation_dot=observation_dot.reshape(observation_dot.shape[0],1)
        # obstacles = mod_obs[6:8]
        # obstacles=obstacles.reshape(obstacles.shape[0],1)
        # obstacles_vel = mod_obs[8:10]
        # obstacles_vel=obstacles_vel.reshape(obstacles_vel.shape[0],1)
        # A = self.compute_A(observation, obstacles)  # For Exercise 1
        # b = self.compute_b(observation, observation_dot,
        #                    obstacles, obstacles_vel)  # For Exercise 1
        # # Starting for umin and umax

        # umin=np.empty((2,1))
        # umax=np.empty((2,1))
        # flag_xmin_ymin=0
        # flag_xmin_ymax=0
        # flag_xmax_ymin=0
        # flag_xmax_ymax=0
        # failure_flag=0
        # def half_plane_interval_intersection(a_1, a_2, b, x_min, x_max, y_min, y_max, flag_xmin_ymax, flag_xmin_ymin, flag_xmax_ymin, flag_xmax_ymax):
        #     """
        #     Find the intersection between a half-plane defined by a_1x + a_2y <= b
        #     and a 2D interval [x_min, x_max] x [y_min, y_max].
        #     """
        #     intersections = []
        #     umin_temp=np.zeros((2,1))
        #     umax_temp=np.zeros((2,1))
        #     # Check if the half-plane intersects the left or right edge of the interval
        #     if a_1 != 0:
        #         x_left = (b - a_2 * y_min) / a_1
        #         x_right = (b - a_2 * y_max) / a_1
        #         if x_min <= x_left <= x_max:
        #             intersections.append((np.double(x_left), np.double(y_min)))
        #             if a_1*(np.double(x_left) - 0.000001)+a_2*np.double(y_min)<=b:
        #                 umin_temp[0]=x_min
        #                 umin_temp[1]=y_min
        #                 flag_xmin_ymin=1
        #             if a_1*(np.double(x_left) + 0.000001)+a_2*np.double(y_min)<=b:
        #                 umax_temp[0]=x_max
        #                 umin_temp[1]=y_min
        #                 flag_xmax_ymin=1
        #         if x_min <= x_right <= x_max:
        #             intersections.append((np.double(x_right), np.double(y_max)))
        #             if a_1*(x_right - 0.000001)+a_2*y_max<=b:
        #                 umin_temp[0]=x_min
        #                 umax_temp[1]=y_max
        #                 flag_xmin_ymax=1
        #             if a_1*(x_right + 0.000001)+a_2*y_max<=b:
        #                 umax_temp[0]=x_max
        #                 umax_temp[1]=y_max
        #                 flag_xmax_ymax=1
        #     # Check if the half-plane intersects the top or bottom edge of the interval
        #     if a_2 != 0:
        #         y_bottom = (b - a_1 * x_min) / a_2
        #         y_top = (b - a_1 * x_max) / a_2
        #         if y_min <= y_bottom <= y_max:
        #             intersections.append((np.double(x_min), np.double(y_bottom)))
        #             if a_1*x_min+a_2*(y_bottom - 0.000001)<=b:
        #                 umin_temp[0]=x_min
        #                 umin_temp[1]=y_min
        #                 flag_xmin_ymin=1
        #             if a_1*x_min+a_2*(y_bottom + 0.000001)<=b:
        #                 umin_temp[0]=x_min
        #                 umax_temp[1]=y_max   
        #                 flag_xmin_ymax=1
        #         if y_min <= y_top <= y_max:
        #             intersections.append((np.double(x_max), np.double(y_top)))
        #             if a_1*x_max+a_2*(y_top - 0.000001)<=b:
        #                 umax_temp[0]=x_max
        #                 umin_temp[1]=y_min
        #                 flag_xmax_ymin=1
        #             if a_1*x_max+a_2*(y_top + 0.000001)<=b:
        #                 umax_temp[0]=x_max
        #                 umax_temp[1]=y_max   
        #                 flag_xmax_ymax=1
        #     intersections_temp=np.array(intersections)
        #     if intersections_temp.shape==(2,2):
        #         intersections=np.array(intersections)
        #     return intersections, flag_xmin_ymax, flag_xmin_ymin, flag_xmax_ymin, flag_xmax_ymax
        # intersections, flag_xmin_ymax, flag_xmin_ymin, flag_xmax_ymin, flag_xmax_ymax = half_plane_interval_intersection(A[0], A[1], b, self.umin[0], self.umax[0], self.umin[1], self.umax[1], flag_xmin_ymax, flag_xmin_ymin, flag_xmax_ymin, flag_xmax_ymax)
        # intersections_temp=np.array(intersections)
        # if intersections_temp.shape!=(2,2):
        #     intersections.append((self.umin[0],self.umin[1]))
        #     intersections.append((self.umax[0],self.umax[1]))
        # #print("the intersection is",intersections)
        
        
        # #Setting up the optimization formulation
        
        # x = cp.Variable(pos=True, name="x")
        # y = cp.Variable(pos=True, name="y")
        # flag_max_min=0
        # flag_max_max=0
        # flag_min_min=0
        # flag_min_max=0
        # flag_none=1
        # if flag_xmin_ymax==1 and flag_xmin_ymin==0 and flag_xmax_ymin==0 and flag_xmax_ymax==0:
        #     #objective=(x-self.umin[0])*(self.umax[1]-y)
        #     flag_min_max=1
        #     flag_none=0
        # elif flag_xmin_ymax==1 and flag_xmin_ymin==1 and flag_xmax_ymin==0 and flag_xmax_ymax==0: #case 2, need slope
        #     #objective=(x-self.umin[0])*(y-self.umin[1])
        #     flag_min_min=1
        #     flag_none=0
            
        # #considering slope for case 2
        # elif flag_xmin_ymax==1 and flag_xmin_ymin==1 and flag_xmax_ymin==0 and flag_xmax_ymax==0 and A[0]*A[1]<0: #case 2, using slope
        #     #objective=(x-self.umin[0])*(self.umax[1]-y)
        #     flag_min_max=1
        #     flag_none=0
        # #done case 2
            
        # elif flag_xmin_ymax==1 and flag_xmax_ymin==1 and flag_xmin_ymin==0 and flag_xmax_ymax==0:
        #     #objective=(x-self.umin[0])*(y-self.umin[1])
        #     flag_min_min=1
        #     flag_none=0
        # elif flag_xmin_ymin==1 and flag_xmax_ymin==1 and flag_xmax_ymax==0 and flag_xmin_ymax==0: #case 4, need slope
        #     #objective=(x-self.umin[0])*(y-self.umin[1])
        #     flag_min_min=1
        #     flag_none=0
            
        # #considering slope for case 4
        # elif flag_xmin_ymin==1 and flag_xmax_ymin==1 and flag_xmax_ymax==0 and flag_xmin_ymax==0 and A[0]*A[1]<0: #case 4, using slope
        #     #objective=(self.umax[0]-x)*(y-self.umin[1])
        #     flag_max_min=1
        #     flag_none=0
        # #done case 4
            
        # elif flag_xmin_ymin==1 and flag_xmin_ymax==0 and flag_xmax_ymin==0 and flag_xmax_ymax==0:
        #     #objective=(x-self.umin[0])*(y-self.umin[1])
        #     flag_min_min=1
        #     flag_none=0
        # elif flag_xmax_ymin==1 and flag_xmax_ymax==0 and flag_xmin_ymin==0 and flag_xmin_ymax==0:
        #     #objective=(self.umax[0]-x)*(y-self.umin[1])
        #     flag_min_min=1
        #     flag_none=0
        # elif flag_xmax_ymax==1 and flag_xmin_ymin==1 and flag_xmax_ymin==0 and flag_xmin_ymax==0:
        #     #objective=(self.umax[0]-x)*(y-self.umin[1])
        #     flag_max_min=1
        #     flag_none=0
        # elif flag_xmin_ymax==1 and flag_xmax_ymin==1 and flag_xmin_ymin==0 and flag_xmax_ymax==0: #case 8, need slope
        #     #objective=(self.umax[0]-x)*(self.umax[1]-y)  
        #     flag_max_max=1
        #     flag_none=0
        
        # #considering slope for case 8
        # elif flag_xmin_ymax==1 and flag_xmax_ymin==1 and flag_xmin_ymin==0 and flag_xmax_ymax==0 and A[0]*A[1]<0: #case 8, using slope
        #     #objective=(self.umax[0]-x)*(y-self.umin[1])
        #     flag_max_min=1
        #     flag_none=0
        # #done case 8            
            
        # elif flag_xmax_ymax==1 and flag_xmax_ymin==0 and flag_xmin_ymin==0 and flag_xmin_ymax==0:
        #     #objective=(self.umax[0]-x)*(self.umax[1]-y)
        #     flag_max_max=1
        #     flag_none=0
        # elif flag_xmin_ymax==1 and flag_xmax_ymax==1 and flag_xmin_ymin==0 and flag_xmax_ymin==0: #case 10, need slope
        #     #objective=(self.umax[0]-x)*(self.umax[1]-y)
        #     flag_max_max=1
        #     flag_none=0
        
        # #considering slope for case 10
        # elif flag_xmin_ymax==1 and flag_xmax_ymax==1 and flag_xmin_ymin==0 and flag_xmax_ymin==0 and A[0]*A[1]<0: #case 10, using slope
        #     #objective=(x-self.umin[0])*(self.umax[1]-y)
        #     flag_min_max=1
        #     flag_none=0
        # #done case 10    
            
        # elif flag_xmin_ymax==1 and flag_xmax_ymin==1 and flag_xmin_ymin==0 and flag_xmax_ymax==0:
        #     #objective=(self.umax[0]-x)*(self.umax[1]-y)
        #     flag_max_max=1
        #     flag_none=0
        # elif flag_xmax_ymax==1 and flag_xmin_ymin==1 and flag_xmax_ymin==0 and flag_xmin_ymax==0:
        #     #objective=(x-self.umin[0])*(self.umax[1]-y)
        #     flag_min_max=1
        #     flag_none=0
        
        # umin_x=copy.deepcopy(self.umin[0])
        # umin_y=copy.deepcopy(self.umin[1])
        # umax_y=copy.deepcopy(self.umax[1])
        # umax_x=copy.deepcopy(self.umax[0])
        # if flag_none==0:    
        #     objective=x*y
        #     epsilon=0.000001
        #     # objective1=(x-self.umin[0])*(y-self.umin[1])
        #     # objective2=(x-self.umin[0])*(self.umax[1]-y)
        #     # objective3=(self.umax[0]-x)*(self.umax[1]-y)
        #     # objective4=(self.umax[0]-x)*(y-self.umin[1])
        #     if flag_min_min==1:
        #         c=np.double(b-A[0]*self.umin[0]-A[1]*self.umin[1])
        #         constraints=[
        #             ((A[0])*x+(A[1])*y)<=c, epsilon<=x, x<=self.umax[0]-self.umin[0], epsilon<=y, y<=self.umax[1]-self.umin[1]] 
        #         def constraint(x):
        #             a1 = A[0]  # coefficient of x
        #             a2 = A[1]  # coefficient of y
        #             return c - a1 * x[0] - a2 * x[1]
        #     elif flag_min_max==1:
        #         c=np.double(b-A[0]*self.umin[0]-A[1]*self.umax[1])
        #         constraints=[
        #             ((A[0])*x+(-A[1])*y)<=c, epsilon<=x, x<=self.umax[0]-self.umin[0], epsilon<=y, y<=self.umax[1]-self.umin[1]]
        #         def constraint(x):
        #             a1 = A[0]  # coefficient of x
        #             a2 = -A[1]  # coefficient of y
        #             return c - a1 * x[0] - a2 * x[1]
        #     elif flag_max_max==1:
        #         c=np.double(b-A[0]*self.umax[0]-A[1]*self.umax[1])
        #         constraints=[
        #             ((-A[0])*x+(-A[1])*y)<=c, epsilon<=x, x<=self.umax[0]-self.umin[0], epsilon<=y, y<=self.umax[1]-self.umin[1]]
        #         def constraint(x):
        #             a1 = -A[0]  # coefficient of x
        #             a2 = -A[1]  # coefficient of y
        #             return c - a1 * x[0] - a2 * x[1]
        #     elif flag_max_min==1:
        #         c=np.double(b-A[0]*self.umax[0]-A[1]*self.umin[1])
        #         constraints=[
        #             ((-A[0])*x+(A[1])*y)<=c, epsilon<=x, x<=self.umax[0]-self.umin[0], epsilon<=y, y<=self.umax[1]-self.umin[1]]
        #         def constraint(x):
        #             a1 = -A[0]  # coefficient of x
        #             a2 = A[1]  # coefficient of y
        #             return c - a1 * x[0] - a2 * x[1]             
        #     try:     
        #         assert objective.is_log_log_concave()
        #         assert all(constraint.is_dgp() for constraint in constraints)
        #         problem = cp.Problem(cp.Maximize(objective), constraints)
        #         #print(problem)
        #         #print("Is this problem DGP?", problem.is_dgp())
        #         problem.solve(gp=True, verbose=False)    
        #         x_opt=x.value
        #         y_opt=y.value
                
        #     #except AssertionError as msg:
        #     except:
        #         #import pdb; pdb.set_trace()
        #         #print("Optimization failed. Message:", msg)
        #         #print(msg)
        #         #print(objective)
        #         #print(constraints)
        #         #problem = cp.Problem(cp.Maximize(objective), constraints)
        #         #print(problem)
        #         #print("Is this problem DGP?", problem.is_dgp())
        #         #problem.solve(gp=True, verbose=False)
        #         # Define the objective function
                
        #         try:
        #             def objective(x):
        #                 return -x[0] * x[1]
    
        #             # Define the initial guess
        #             x0 = np.array([self.x_opt, self.y_opt])
    
        #             # Define the bounds for the variables
        #             #bounds = [(None, None), (None, None)]  # No specific bounds for x and y
        #             #bounds = [(0, 10), (0, 10)]
        #             bounds=[(0, self.umax[0]-self.umin[0]), (0, self.umax[1]-self.umin[1])]
        #             # Solve the optimization problem
                    
        #             methods_list = ['trust-constr','SLSQP','Nelder-Mead','Powell','CG','BFGS','Newton-CG',
        #                            'L-BFGS-B','TNC','COBYLA','dogleg','trust-ncg',
        #                            'trust-exact','trust-krylov']
        #             for methods in methods_list:
        #                 result = minimize(objective, x0, method=methods, constraints={'type': 'ineq', 'fun': constraint}, bounds=bounds)
        #                 if result.success:
        #                     # Retrieve the optimal values
        #                     x_opt = result.x[0]
        #                     y_opt = result.x[1]
        #                     break
        #                     #print("Optimal values:")
        #                     #print("x =", x_opt)
        #                     #print("y =", y_opt)
        #                 else:
        #                     #import pdb; pdb.set_trace()
        #                     failure_flag=1
        #                     print("Optimization failed for all the solvers. Message:", result.message)
        #                     print(A)
        #                     print(b)
        #                     print(observation)
        #                     print(observation_dot)
        #                     #print("Optimization failed. Message:", result.message)
        #                     x_opt=self.x_opt
        #                     y_opt=self.y_opt #randomly assigning -09.02.23
        #         except:
        #             #import pdb; pdb.set_trace()
        #             failure_flag=1
        #             print(A)
        #             print(b)
        #             print(observation)
        #             print(observation_dot)
        #             print("Second Exception Failed!! Optimization failed for everything. Message:", result.message)
        #             x_opt=self.x_opt
        #             y_opt=self.y_opt #randomly assigning -09.02.23
        #             #import pdb; pdb.set_trace()
                    
        #     self.x_opt=copy.deepcopy(x_opt)
        #     self.y_opt=copy.deepcopy(y_opt)
            
        #     if flag_min_min==1:
        #         x_true=x_opt+self.umin[0]
        #         y_true=y_opt+self.umin[1]
        #         umin[0]=umin_x
        #         umin[1]=umin_y
        #         umax[0]=x_true
        #         umax[1]=y_true
        #     elif flag_min_max==1:
        #         x_true=x_opt+self.umin[0]
        #         y_true=self.umax[1]-y_opt
        #         umin[0]=umin_x
        #         umin[1]=y_true
        #         umax[0]=x_true
        #         umax[1]=umax_y
        #     elif flag_max_max==1:
        #         x_true=self.umax[0]-x_opt
        #         y_true=self.umax[1]-y_opt
        #         umin[0]=x_true
        #         umin[1]=y_true
        #         umax[0]=umax_x
        #         umax[1]=umax_y
        #     elif flag_max_min==1:
        #         x_true=self.umax[0]-x_opt
        #         y_true=y_opt+self.umin[1]
        #         umin[0]=x_true
        #         umin[1]=umin_y
        #         umax[0]=umax_x
        #         umax[1]=y_true
        # if flag_none==1:
        #     umin[0]=umin_x
        #     umin[1]=umin_y
        #     umax[0]=umax_x
        #     umax[1]=umax_y
        #     #print("this case also got triggered")
            
        # #print("the optimal_interval is",np.array([[umin[0],umax[0]],
        #                                           #[umin[1],umax[1]]]))
        
        # # umin[0]=np.double(intersections[0][0])
        # # umin[1]=np.double(intersections[0][1])
        # # umax[0]=np.double(intersections[1][0])
        # # umax[1]=np.double(intersections[1][1])
        # """
        # def maximal_hyperrectangle(a1, a2, b, interval):
        #     # Find the range of x and y values in the interval
        #     x_min, x_max = interval[0][0], interval[0][1]
        #     y_min, y_max = interval[1][0], interval[1][1]
            
        #     # Initialize the maximum interval
        #     max_interval = None
            
        #     # Loop over all possible x and y values in the interval
        #     for x in np.linspace(x_min, x_max, num=100):
        #         for y in np.linspace(y_min, y_max, num=100):
        #             # Check if the point (x, y) satisfies the inequality
        #             if a1*x + a2*y <= b:
        #                 # If this is the first valid point, initialize the maximum interval
        #                 if max_interval is None:
        #                     max_interval = [(x, x), (y, y)]
        #                 # Otherwise, update the maximum interval to include this point
        #                 else:
        #                     max_interval[0] = (min(max_interval[0][0], x), max(max_interval[0][1], x))
        #                     max_interval[1] = (min(max_interval[1][0], y), max(max_interval[1][1], y))
            
        #     # Return the maximum interval
        #     return max_interval
        # interval = [(self.umin[0], self.umax[0]), (self.umin[1], self.umax[1])]
        # max_interval = maximal_hyperrectangle(A[0], A[1], b, interval)
        # print("the max_interval is",max_interval)
        # #print(max_interval)
        # umin[0]=np.double(max_interval[0][0])
        # umax[0]=np.double(max_interval[0][1])
        # umin[1]=np.double(max_interval[1][0])
        # umax[1]=np.double(max_interval[1][1])
        # """
        # if failure_flag ==1 and (self._get_obs()[0] - self.obstacle[0])**4/(self.a_d)**4 + (self._get_obs()[1] - self.obstacle[1])**4/(self.b_d)**4 < 1.5*self.safety_dist:
        #     self.failure_flag=1
        #     umin=0.0*self.umin
        #     umax=0.0*self.umax
        umax=self.umax.flatten()
        umin=self.umin.flatten()
        return [umin, umax]

    def cbf_proj(self, mod_obs):  # cbf for a quadrotor with static obstacle(s)
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
        A = self.compute_A(observation, obstacles)  # For Exercise 1
        b = self.compute_b(observation, observation_dot,
                           obstacles, obstacles_vel)  # For Exercise 1
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
                    if a_1*(np.double(x_left) - 0.000001)+a_2*np.double(y_min)<=b:
                        umin_temp[0]=x_min
                        umin_temp[1]=y_min
                        flag_xmin_ymin=1
                    if a_1*(np.double(x_left) + 0.000001)+a_2*np.double(y_min)<=b:
                        umax_temp[0]=x_max
                        umin_temp[1]=y_min
                        flag_xmax_ymin=1
                if x_min <= x_right <= x_max:
                    intersections.append((np.double(x_right), np.double(y_max)))
                    if a_1*(x_right - 0.000001)+a_2*y_max<=b:
                        umin_temp[0]=x_min
                        umax_temp[1]=y_max
                        flag_xmin_ymax=1
                    if a_1*(x_right + 0.000001)+a_2*y_max<=b:
                        umax_temp[0]=x_max
                        umax_temp[1]=y_max
                        flag_xmax_ymax=1
            # Check if the half-plane intersects the top or bottom edge of the interval
            if a_2 != 0:
                y_bottom = (b - a_1 * x_min) / a_2
                y_top = (b - a_1 * x_max) / a_2
                if y_min <= y_bottom <= y_max:
                    intersections.append((np.double(x_min), np.double(y_bottom)))
                    if a_1*x_min+a_2*(y_bottom - 0.000001)<=b:
                        umin_temp[0]=x_min
                        umin_temp[1]=y_min
                        flag_xmin_ymin=1
                    if a_1*x_min+a_2*(y_bottom + 0.000001)<=b:
                        umin_temp[0]=x_min
                        umax_temp[1]=y_max   
                        flag_xmin_ymax=1
                if y_min <= y_top <= y_max:
                    intersections.append((np.double(x_max), np.double(y_top)))
                    if a_1*x_max+a_2*(y_top - 0.000001)<=b:
                        umax_temp[0]=x_max
                        umin_temp[1]=y_min
                        flag_xmax_ymin=1
                    if a_1*x_max+a_2*(y_top + 0.000001)<=b:
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
        #print("the intersection is",intersections)
        
        
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
            #objective=(x-self.umin[0])*(y-self.umin[1])
            flag_min_min=1
            flag_none=0
            
        #considering slope for case 2
        elif flag_xmin_ymax==1 and flag_xmin_ymin==1 and flag_xmax_ymin==0 and flag_xmax_ymax==0 and A[0]*A[1]<0: #case 2, using slope
            #objective=(x-self.umin[0])*(self.umax[1]-y)
            flag_min_max=1
            flag_none=0
        #done case 2
            
        elif flag_xmin_ymax==1 and flag_xmax_ymin==1 and flag_xmin_ymin==0 and flag_xmax_ymax==0:
            #objective=(x-self.umin[0])*(y-self.umin[1])
            flag_min_min=1
            flag_none=0
        elif flag_xmin_ymin==1 and flag_xmax_ymin==1 and flag_xmax_ymax==0 and flag_xmin_ymax==0: #case 4, need slope
            #objective=(x-self.umin[0])*(y-self.umin[1])
            flag_min_min=1
            flag_none=0
            
        #considering slope for case 4
        elif flag_xmin_ymin==1 and flag_xmax_ymin==1 and flag_xmax_ymax==0 and flag_xmin_ymax==0 and A[0]*A[1]<0: #case 4, using slope
            #objective=(self.umax[0]-x)*(y-self.umin[1])
            flag_max_min=1
            flag_none=0
        #done case 4
            
        elif flag_xmin_ymin==1 and flag_xmin_ymax==0 and flag_xmax_ymin==0 and flag_xmax_ymax==0:
            #objective=(x-self.umin[0])*(y-self.umin[1])
            flag_min_min=1
            flag_none=0
        elif flag_xmax_ymin==1 and flag_xmax_ymax==0 and flag_xmin_ymin==0 and flag_xmin_ymax==0:
            #objective=(self.umax[0]-x)*(y-self.umin[1])
            flag_min_min=1
            flag_none=0
        elif flag_xmax_ymax==1 and flag_xmin_ymin==1 and flag_xmax_ymin==0 and flag_xmin_ymax==0:
            #objective=(self.umax[0]-x)*(y-self.umin[1])
            flag_max_min=1
            flag_none=0
        elif flag_xmin_ymax==1 and flag_xmax_ymin==1 and flag_xmin_ymin==0 and flag_xmax_ymax==0: #case 8, need slope
            #objective=(self.umax[0]-x)*(self.umax[1]-y)  
            flag_max_max=1
            flag_none=0
        
        #considering slope for case 8
        elif flag_xmin_ymax==1 and flag_xmax_ymin==1 and flag_xmin_ymin==0 and flag_xmax_ymax==0 and A[0]*A[1]<0: #case 8, using slope
            #objective=(self.umax[0]-x)*(y-self.umin[1])
            flag_max_min=1
            flag_none=0
        #done case 8            
            
        elif flag_xmax_ymax==1 and flag_xmax_ymin==0 and flag_xmin_ymin==0 and flag_xmin_ymax==0:
            #objective=(self.umax[0]-x)*(self.umax[1]-y)
            flag_max_max=1
            flag_none=0
        elif flag_xmin_ymax==1 and flag_xmax_ymax==1 and flag_xmin_ymin==0 and flag_xmax_ymin==0: #case 10, need slope
            #objective=(self.umax[0]-x)*(self.umax[1]-y)
            flag_max_max=1
            flag_none=0
        
        #considering slope for case 10
        elif flag_xmin_ymax==1 and flag_xmax_ymax==1 and flag_xmin_ymin==0 and flag_xmax_ymin==0 and A[0]*A[1]<0: #case 10, using slope
            #objective=(x-self.umin[0])*(self.umax[1]-y)
            flag_min_max=1
            flag_none=0
        #done case 10    
            
        elif flag_xmin_ymax==1 and flag_xmax_ymin==1 and flag_xmin_ymin==0 and flag_xmax_ymax==0:
            #objective=(self.umax[0]-x)*(self.umax[1]-y)
            flag_max_max=1
            flag_none=0
        elif flag_xmax_ymax==1 and flag_xmin_ymin==1 and flag_xmax_ymin==0 and flag_xmin_ymax==0:
            #objective=(x-self.umin[0])*(self.umax[1]-y)
            flag_min_max=1
            flag_none=0
        
        umin_x=copy.deepcopy(self.umin[0])
        umin_y=copy.deepcopy(self.umin[1])
        umax_y=copy.deepcopy(self.umax[1])
        umax_x=copy.deepcopy(self.umax[0])
        if flag_none==0:    
            objective=x*y
            epsilon=0.000001
            # objective1=(x-self.umin[0])*(y-self.umin[1])
            # objective2=(x-self.umin[0])*(self.umax[1]-y)
            # objective3=(self.umax[0]-x)*(self.umax[1]-y)
            # objective4=(self.umax[0]-x)*(y-self.umin[1])
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
                #print(problem)
                #print("Is this problem DGP?", problem.is_dgp())
                problem.solve(gp=True, verbose=False)    
                x_opt=x.value
                y_opt=y.value
                
            #except AssertionError as msg:
            except:
                #import pdb; pdb.set_trace()
                #print("Optimization failed. Message:", msg)
                #print(msg)
                #print(objective)
                #print(constraints)
                #problem = cp.Problem(cp.Maximize(objective), constraints)
                #print(problem)
                #print("Is this problem DGP?", problem.is_dgp())
                #problem.solve(gp=True, verbose=False)
                # Define the objective function
                
                try:
                    def objective(x):
                        return -x[0] * x[1]
    
                    # Define the initial guess
                    x0 = np.array([self.x_opt, self.y_opt])
    
                    # Define the bounds for the variables
                    #bounds = [(None, None), (None, None)]  # No specific bounds for x and y
                    #bounds = [(0, 10), (0, 10)]
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
                            #print("Optimal values:")
                            #print("x =", x_opt)
                            #print("y =", y_opt)
                        else:
                            #import pdb; pdb.set_trace()
                            failure_flag=1
                            print("Optimization failed for all the solvers. Message:", result.message)
                            print(A)
                            print(b)
                            print(observation)
                            print(observation_dot)
                            #print("Optimization failed. Message:", result.message)
                            x_opt=self.x_opt
                            y_opt=self.y_opt #randomly assigning -09.02.23
                except:
                    #import pdb; pdb.set_trace()
                    failure_flag=1
                    print(A)
                    print(b)
                    print(observation)
                    print(observation_dot)
                    print("Second Exception Failed!! Optimization failed for everything. Message:", result.message)
                    x_opt=self.x_opt
                    y_opt=self.y_opt #randomly assigning -09.02.23
                    #import pdb; pdb.set_trace()
                    
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
            #print("this case also got triggered")
            
        #print("the optimal_interval is",np.array([[umin[0],umax[0]],
                                                  #[umin[1],umax[1]]]))
        
        # umin[0]=np.double(intersections[0][0])
        # umin[1]=np.double(intersections[0][1])
        # umax[0]=np.double(intersections[1][0])
        # umax[1]=np.double(intersections[1][1])
        """
        def maximal_hyperrectangle(a1, a2, b, interval):
            # Find the range of x and y values in the interval
            x_min, x_max = interval[0][0], interval[0][1]
            y_min, y_max = interval[1][0], interval[1][1]
            
            # Initialize the maximum interval
            max_interval = None
            
            # Loop over all possible x and y values in the interval
            for x in np.linspace(x_min, x_max, num=100):
                for y in np.linspace(y_min, y_max, num=100):
                    # Check if the point (x, y) satisfies the inequality
                    if a1*x + a2*y <= b:
                        # If this is the first valid point, initialize the maximum interval
                        if max_interval is None:
                            max_interval = [(x, x), (y, y)]
                        # Otherwise, update the maximum interval to include this point
                        else:
                            max_interval[0] = (min(max_interval[0][0], x), max(max_interval[0][1], x))
                            max_interval[1] = (min(max_interval[1][0], y), max(max_interval[1][1], y))
            
            # Return the maximum interval
            return max_interval
        interval = [(self.umin[0], self.umax[0]), (self.umin[1], self.umax[1])]
        max_interval = maximal_hyperrectangle(A[0], A[1], b, interval)
        print("the max_interval is",max_interval)
        #print(max_interval)
        umin[0]=np.double(max_interval[0][0])
        umax[0]=np.double(max_interval[0][1])
        umin[1]=np.double(max_interval[1][0])
        umax[1]=np.double(max_interval[1][1])
        """
        if failure_flag ==1 and (self._get_obs()[0] - self.obstacle[0])**4/(self.a_d)**4 + (self._get_obs()[1] - self.obstacle[1])**4/(self.b_d)**4 < 1.3*self.safety_dist:
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

    # majoor :) updates in this obstacle update function
    # currently it can take in multiple STATIC obstacles
    # for ONE robot --vipul

    def update_obstacles(self, noisy=False):
        obstacle = []
        obstacle_vel = []
        # checking whether static obstacles are there through obs array   --VIPUL
        if not len(self.obstacle):
            # return {"obstacle":obstacle, "obstacle_vel":obstacle_vel}
            return self.obstacles, self.obstacles_vel
        # return if no static obstacle found --vipul

        # else register them --vipul
        if self.num_obstacles == 1:  # for single obstacle --vipul
            obstacle.append(self.obstacle.reshape(2, 1))
            self.obstacle = np.asarray(obstacle).reshape(2, 1)
            obstacle_vel.append(np.array([0, 0]))
            self.obstacle_vel = np.asarray(obstacle_vel).reshape(2, 1)
            # return {"obstacle":obstacle, "obstacle_vel":obstacle_vel} #original--vipul
            return self.obstacle, self.obstacle_vel
        # for multiple obstacles --vipul
        for i in range(self.obstacle.shape[0]):
            self.obstacle.append(self.obstacle[i].reshape(2, 1))
            self.obstacle_vel.append(np.array([0, 0]))

        #obstacles = {"obstacle":obstacle, "obstacle_vel":obstacle_vel}
        return self.obstacle, self.obstacle_vel

    @np.vectorize
    def h_func(r1, r2, a, b, safety_dist):
        hr = np.power(r1, 4)/np.power(a, 4) + \
            np.power(r2, 4)/np.power(b, 4) - safety_dist
        return hr

    def plot_step(self, new_obs, u_hat_acc, state_hist, plot_handle):  # need some more work--vipul
        state_hist_plot = np.array(state_hist)
        nom_cont = self.compute_nom_control()
        multiplier_const = 15
        plot_handle.plot([state_hist_plot[-1, 0], state_hist_plot[-1, 0] + multiplier_const *
                          u_hat_acc[0]],
                         [state_hist_plot[-1, 1], state_hist_plot[-1, 1] + multiplier_const * u_hat_acc[1]], label="Safe", color='b')
        plot_handle.plot([state_hist_plot[-1, 0], state_hist_plot[-1, 0] + multiplier_const *
                          nom_cont[0]],
                         [state_hist_plot[-1, 1], state_hist_plot[-1, 1] + multiplier_const * nom_cont[1]], label="Nominal", color='orange')

        plot_handle.plot(state_hist_plot[:, 0], state_hist_plot[:, 1])
        plot_handle.plot(self.goal[0], self.goal[1], '*r')
        # modification --vipul
        plot_handle.text(self.goal[0]+0.2, self.goal[1]+0.2, color='r')
        # plot_handle.plot(state_hist_plot[-1, 0], state_hist_plot[-1, 1], '8k') # current
        # modification --vipul
        plot_handle.text(
            state_hist_plot[-1, 0]+0.2, state_hist_plot[-1, 1]+0.2)
        if self.is_crash:
            plot_handle.set_title("CRASHED!")

        for i in range(new_obs.shape[1]):
            plot_handle.plot(new_obs[0, i], new_obs[1, i], '8k')  # obs

        ell = Ellipse((state_hist_plot[-1, 0], state_hist_plot[-1, 1]),
                      self.a_d*self.safety_dist+0.5, self.b_d*self.safety_dist+0.5, 0)
        ell.set_alpha(0.3)
        ell.set_facecolor(np.array([0, 1, 0]))

        plot_handle.add_artist(ell)

        ell = Ellipse((state_hist_plot[-1, 0], state_hist_plot[-1, 1]),
                      self.robot_radius+0.5, self.robot_radius+0.5, 0)
        ell.set_alpha(0.8)
        ell.set_facecolor(np.array([1, 0, 0]))

        plot_handle.add_artist(ell)

        plot_handle.set_xlim([-10, 10])
        plot_handle.set_ylim([-10, 10])

    def solve_qp(P, q, G, h):
        # Custom wrapper cvxopt.solvers.qp
        # Takes in numpy array Converts to matrix double
        P = matrix(P, tc='d')
        q = matrix(q, tc='d')
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')
        solvers.options['show_progress'] = False
        Sol = solvers.qp(P, q, G, h)

        return Sol

    # copying functions from controller.py

    def go_to_acceleration(self, des_acc):  # mofified --vipul
        # pass
        des_theta, des_thrust_pc = self.dynamic_inversion(des_acc)
        u = self.pi_attitude_control(
            des_theta, des_thrust_pc)  # attitude control
        return u

    def dynamic_inversion(self, des_acc):  # modified --vipul
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
        # tot_u_constant = 408750 * 4  # hover, for four motors
        # specific_force = tot_u_constant  / param_dict["m"]

        # based on http://research.sabanciuniv.edu/33398/1/ICUAS2017_Final_ZAKI_UNEL_YILDIZ.pdf (Eq. 22-24)
        U1 = np.linalg.norm(des_acc - np.array([0, 0, self.param_dict["g"]]))
        des_pitch_noyaw = np.arcsin(des_acc[0] / U1)
        des_angle = [des_pitch_noyaw,
                     np.arcsin(des_acc[1] / (U1 * np.cos(des_pitch_noyaw)))]
        des_pitch = des_angle[0] * np.cos(yaw) + des_angle[1] * np.sin(yaw)
        des_roll = des_angle[0] * np.sin(yaw) - des_angle[1] * np.cos(yaw)

        # TODO: move to attitude controller?
        des_pitch = np.clip(des_pitch, np.radians(-30), np.radians(30))
        des_roll = np.clip(des_roll, np.radians(-30), np.radians(30))

        # TODO: currently, set yaw as constant
        des_yaw = yaw
        des_theta = [des_roll, des_pitch, des_yaw]

        # vertical (acc_z -> thrust)

        thrust = (self.param_dict["m"] * (des_acc[2] -
                                          self.param_dict["g"]))/self.param_dict["k"]  # T=ma/k
        max_tot_u = 400000000.0  # TODO: make in param_dict
        des_thrust_pc = thrust/max_tot_u

        return des_theta, des_thrust_pc

    def go_to_position(self, des_pos, integral_p_err=None, integral_v_err=None):  # modified--vipul

        des_vel, integral_p_err = self.pi_position_control(
            des_pos, integral_p_err)
        des_thrust, des_theta, integral_v_err = self.pi_velocity_control(
            des_vel, integral_v_err)  # attitude control
        # des_theta_deg = np.degrees(des_theta) # for logging
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

        # Compute total u
        # tot_thrust = (m * g) / (k * np.cos(theta[1]) * np.cos(theta[0])) # more like tot base u
        # print("tot_thrust", tot_thrust)
        max_tot_u = 400000000.0
        tot_u = des_thrust_pc * max_tot_u

        # Compute errors
        # TODO: set thetadot to zero?
        e = Kd * thetadot + Kp * (theta - des_theta)
        # print("e_theta", e)

        # Compute control input given angular error (dynamic inversion)
        u = self.angerr2u(e, theta, tot_u)
        return u

    def wrap2pi(self, ang_diff):
        """For angle difference."""
        while ang_diff > np.pi/2 or ang_diff < -np.pi/2:
            if ang_diff > np.pi/2:
                ang_diff -= np.pi
            else:
                ang_diff += np.pi

        return ang_diff

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

    # copying functions from visualize_dynamics

    def visualize_quad_quadhist(self, ax, quad_hist, t):
        """Works with QuadHist class."""
        self.visualize_quad(ax, quad_hist.hist_x[:t], quad_hist.hist_y[:t],
                            quad_hist.hist_z[:t], quad_hist.hist_pos[t], quad_hist.hist_theta[t])

    def visualize_error_quadhist(self, ax_x_error, ax_xd_error, ax_th_error, ax_thr_error, ax_xdd_error, quad_hist, t, dt):
        """Works with QuadHist class."""
        self.visualize_error(ax_x_error, ax_xd_error, ax_th_error, ax_thr_error, ax_xdd_error,
                             quad_hist.hist_pos[:t+1], quad_hist.hist_xdot[:t+1], quad_hist.hist_theta[:t+1], quad_hist.hist_des_theta[:t +
                                                                                                                                       1], quad_hist.hist_thetadot[:t+1], dt, quad_hist.hist_des_xdot[:t+1], quad_hist.hist_des_x[:t+1],
                             quad_hist.hist_xdotdot[:t+1])

    def animate_quad(self, ax, hist_x, hist_y, hist_z, cur_state, cur_theta):
        """Plot quadrotor 3D position and history"""
        x = cur_state
        theta = np.radians(cur_theta)
        R = self.get_rot_matrix(theta)
        plot_L = 1
        quad_ends_body = np.array(
            [[-plot_L, 0, 0], [plot_L, 0, 0], [0, -plot_L, 0], [0, plot_L, 0], [0, 0, 0], [0, 0, 0]]).T
        quad_ends_world = np.dot(R, quad_ends_body) + \
            np.matlib.repmat(x, 6, 1).T
        # Plot Rods
        ax.plot3D(quad_ends_world[0, 0:2],
                  quad_ends_world[1, 0:2], quad_ends_world[2, 0:2], 'r')
        ax.plot3D(quad_ends_world[0, 2:4],
                  quad_ends_world[1, 2:4], quad_ends_world[2, 2:4], 'b')
        # Plot drone center
        ax.scatter3D(x[0], x[1], x[2], edgecolor="r", facecolor="r")

        # Plot history
        ax.scatter3D(hist_x, hist_y, hist_z, edgecolor="b",
                     facecolor="b", alpha=0.1)
        # ax_th_error.
        ax.set_xlim(x[0]-3, x[0]+3)
        ax.set_ylim(x[1]-3, x[1]+3)
        ax.set_zlim(x[2]-5, x[2]+5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        self.plt.pause(0.1)

    def visualize_quad(self, ax, hist_x, hist_y, hist_z, cur_state, cur_theta):
        """Plot quadrotor 3D position and history"""
        x = cur_state
        theta = np.radians(cur_theta)
        R = self.get_rot_matrix(theta)
        plot_L = 1
        quad_ends_body = np.array(
            [[-plot_L, 0, 0], [plot_L, 0, 0], [0, -plot_L, 0], [0, plot_L, 0], [0, 0, 0], [0, 0, 0]]).T
        quad_ends_world = np.dot(R, quad_ends_body) + \
            np.matlib.repmat(x, 6, 1).T
        # Plot Rods
        ax.plot3D(quad_ends_world[0, [1, 5]],
                  quad_ends_world[1, [1, 5]], quad_ends_world[2, [1, 5]], 'r')  # body x front
        ax.plot3D(quad_ends_world[0, [0, 5]],
                  quad_ends_world[1, [0, 5]], quad_ends_world[2, [0, 5]], 'k')  # body x back
        # ax.plot3D(quad_ends_world[0, 0:2],
        #           quad_ends_world[1, 0:2], quad_ends_world[2, 0:2], 'r') # body x
        ax.plot3D(quad_ends_world[0, 2:4],
                  quad_ends_world[1, 2:4], quad_ends_world[2, 2:4], 'b')  # body y
        # Plot drone center
        ax.scatter3D(x[0], x[1], x[2], edgecolor="r", facecolor="r")

        # Plot history
        ax.scatter3D(hist_x, hist_y, hist_z, edgecolor="b",
                     facecolor="b", alpha=0.1)
        # ax_th_error.
        ax.set_xlim(x[0]-3, x[0]+3)
        ax.set_ylim(x[1]-3, x[1]+3)
        ax.set_zlim(x[2]-5, x[2]+5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        self.plt.pause(0.1)

    def visualize_error(self, ax_x_error, ax_xd_error, ax_th_error, ax_thr_error, ax_xdd_error, hist_pos, hist_xdot, hist_theta, hist_des_theta, hist_thetadot, dt, hist_des_xdot, hist_des_x, hist_xdotdot):
        # pass
        # ax.plot([0,1], [1,10],'b')

        # Position Error
        ax_x_error.plot(np.array(range(len(hist_theta))) *
                        dt, np.array(hist_pos)[:, 0], 'k')
        ax_x_error.plot(np.array(range(len(hist_theta))) *
                        dt, np.array(hist_pos)[:, 1], 'b')
        ax_x_error.plot(np.array(range(len(hist_theta))) *
                        dt, np.array(hist_pos)[:, 2], 'r')
        # Desired Pos
        ax_x_error.plot(np.array(range(len(hist_theta))) *
                        dt, np.array(hist_des_x)[:, 0], 'k--')
        ax_x_error.plot(np.array(range(len(hist_theta))) *
                        dt, np.array(hist_des_x)[:, 1], 'b--')
        ax_x_error.plot(np.array(range(len(hist_theta))) *
                        dt, np.array(hist_des_x)[:, 2], 'r--')
        ax_x_error.set_title("Position (world)")
        ax_x_error.legend(["x", "y", "z"])

        # TODO: make into funciton for each plot
        # Velocity Error
        ax_xd_error.plot(np.array(range(len(hist_theta))) *
                         dt, np.array(hist_xdot)[:, 0], 'k')
        ax_xd_error.plot(np.array(range(len(hist_theta))) *
                         dt, np.array(hist_xdot)[:, 1], 'b')
        ax_xd_error.plot(np.array(range(len(hist_theta))) *
                         dt, np.array(hist_xdot)[:, 2], 'r')
        # Desired Velocity
        ax_xd_error.plot(np.array(range(len(hist_theta))) *
                         dt, np.array(hist_des_xdot)[:, 0], 'k--')
        ax_xd_error.plot(np.array(range(len(hist_theta))) *
                         dt, np.array(hist_des_xdot)[:, 1], 'b--')
        ax_xd_error.plot(np.array(range(len(hist_theta))) *
                         dt, np.array(hist_des_xdot)[:, 2], 'r--')
        ax_xd_error.legend(["x", "y", "z"])
        ax_xd_error.set_title("Velocity (world)")

        # Angle Error
        ax_th_error.plot(np.array(range(len(hist_theta))) *
                         dt, np.array(hist_theta)[:, 0], 'k')
        ax_th_error.plot(np.array(range(len(hist_theta))) *
                         dt, np.array(hist_theta)[:, 1], 'b')
        ax_th_error.plot(np.array(range(len(hist_theta))) *
                         dt, np.array(hist_theta)[:, 2], 'r')
        # Desired angle
        ax_th_error.plot(np.array(range(len(hist_theta))) *
                         dt, np.array(hist_des_theta)[:, 0], 'k--')
        ax_th_error.plot(np.array(range(len(hist_theta))) *
                         dt, np.array(hist_des_theta)[:, 1], 'b--')
        ax_th_error.plot(np.array(range(len(hist_theta))) *
                         dt, np.array(hist_des_theta)[:, 2], 'r--')

        ax_th_error.legend(["Roll", "Pitch", "Yaw"])
        ax_th_error.set_ylim(-40, 40)
        ax_th_error.set_title("Angle")

        # Angle Rate
        ax_thr_error.plot(np.array(range(len(hist_theta))) *
                          dt, np.array(hist_thetadot)[:, 0], 'k')
        ax_thr_error.plot(np.array(range(len(hist_theta))) *
                          dt, np.array(hist_thetadot)[:, 1], 'b')
        ax_thr_error.plot(np.array(range(len(hist_theta))) *
                          dt, np.array(hist_thetadot)[:, 2], 'r')
        # ax.plot(range(len(hist_theta)), np.array(des_theta)[:, 0])
        ax_thr_error.legend(["Roll Rate", "Pitch Rate", "Yaw Rate"])
        ax_thr_error.set_ylim(-100, 100)

        ax_thr_error.set_title("Angular Rate")

        # Acceleration
        ax_xdd_error.plot(np.array(range(len(hist_theta))) *
                          dt, np.array(hist_xdotdot)[:, 0], 'k')
        ax_xdd_error.plot(np.array(range(len(hist_theta))) *
                          dt, np.array(hist_xdotdot)[:, 1], 'b')
        ax_xdd_error.plot(np.array(range(len(hist_theta))) *
                          dt, np.array(hist_xdotdot)[:, 2], 'r')
        ax_xdd_error.legend(["x", "y", "z"])
        ax_xdd_error.set_title("Acc. (world)")

        self.plt.pause(0.1)
        
#older umin and umax calculations
        #umax = self.umax
        #umax=np.multiply(np.linalg.pinv(A),b)
        #umin = self.umin   
        # if A[0] > 0 and A[1] > 0 and b>=0:
        #     #umax[0] = (b-A[1]*self.umax[1])/A[0]; #intersection with axis u[0]
        #     #umax[0] = max(0,min(self.umax[0], b/A[0]))/2
        #     umax[0] = min(self.umax[0], b/A[0])/2
        #     #umax[1] = (b-A[0]*self.umax[0])/A[1]; #intersection with axis \u[1]
        #     #umax[1] = max(0,min(self.umax[1], b/A[1]))/2
        #     umax[1] = min(self.umax[1], b/A[1])/2
        # if A[0] < 0 and A[1] > 0 and b>=0:
        #     umax[0] = self.umax[0]
        #     umax[1] = min(self.umax[1],b/A[1])
        #     #print("Fix this by writing the appropriate code!!")
        # if A[0] < 0 and A[1] < 0 and b>=0:
        #     umax[0] = self.umax[0]
        #     umax[1] = self.umax[1]
        # if A[0] > 0 and A[1] < 0 and b>=0:
        #     umax[0] = min(self.umax[0],b/A[0])
        #     umax[1] = self.umax[1]  
        # if A[0] == 0 and A[1] > 0 and b>=0:
        #     umax[0] = self.umax[0]
        #     umax[1] = min(self.umax[1],b/A[1])
        # if A[0] > 0 and A[1] == 0 and b>=0:
        #     umax[0] = min(self.umax[0],b/A[0])
        #     umax[1] = self.umax[1]
        # if A[0] ==0 and A[1] == 0 and b>0:
        #     umax[0] = self.umax[0]
        #     umax[1] = self.umax[1]
        # if A[0] > 0 and A[1] > 0 and b<0:
        #     umax = np.zeros(2)
        # if A[0] < 0 and A[1] > 0 and b<0:
        #     umin[0] = min(abs(b)/abs(A[0]),self.umax[0])
        #     umax[1] = 0
        #     umax[0] = self.umax[0]
        # if A[0] > 0 and A[1] < 0 and b<0:
        #     umin[1] = min(abs(b)/abs(A[0]),self.umax[0])
        #     umax[0] = 0
        #     umax[1] = self.umax[1]
        # if umax[0]==0 and umax[1]==0: 
        #     print("this case is left")
        #     #print("Fix this by writing the appropriate code!!")
        # # for i in range(umax.shape[0]):
        # #     if umax[i] > 0:
        # #        umin[i] = -umax[i]
        # #     else:
        # #         umax[i] = -umax[i]
        # #         umin[i] = -umax[i]        
