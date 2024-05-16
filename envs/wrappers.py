#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import mujoco


_FLOAT_EPS = np.finfo(np.float64).eps

def quat2mat(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))

class AntContactsWrapper(gym.Wrapper):
    def __init__(self, env, task, desired_speed=1.0, include_contacts=True,mod_done=True,no_done=False):
        super().__init__(env)
        self.env = env
        self.task = task
        self.include_contacts = include_contacts
        if not(self.include_contacts):
            obs = self._get_obs()
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,shape=(len(obs),))
        # upside down if both boundaries are violated
        self._upright_theta= 0.35
        self._unhealthy_theta_range = np.pi - self._upright_theta
        self._healthy_theta_boundary = 2.7
        self._upside_down_start_time = None
        self._stuck_thresh = 1.
        self._desired_speed = desired_speed
        self.mod_done = mod_done
        self.no_done = no_done

    def _update_contacts(self):
        # with mujoco200 + mujoco_py >= 2.0.0, need to calculate contacts
        mujoco.mj_rnePostConstraint(self.env.sim.model, self.env.sim.data)

    def _done(self,state):
        quat = state[3:7]
        zz = quat2mat(quat)[2,2]
        theta = np.arccos(zz)
        z = state[2]

        # check if agent is stuck upside down
        is_upside_down =  (theta > self._healthy_theta_boundary) 
        duration = 0.
        if is_upside_down:
            current_time = self.env.sim.data.time
            if self._upside_down_start_time is None:
                self._upside_down_start_time = current_time
            else:
                duration = current_time - self._upside_down_start_time
        else:
            self._upside_down_start_time = None
        is_healthy = np.isfinite(state).all() and (duration < self._stuck_thresh)
        return not is_healthy

    def _upright(self,state):
        # 0 = upright, -1 = upside down, decay between _upright_theta and pi
        quat = state[3:7]
        zz = quat2mat(quat)[2,2]
        theta = np.arccos(zz)

        if theta > self._upright_theta:
            err = np.square( (self._upright_theta - theta) / self._unhealthy_theta_range )
        else:
            err = 0.
        return -err

    def step(self, action):
        # 1. run general step
        _, reward, done, info = self.env.step(action) # reward = (xposafter - xposbefore) / self.dt
        # 2. upate contacts
        self._update_contacts()
        # 3. fix observation to account for contacts (or lack of contacts)
        new_obs = self._get_obs()
        # 4. update reward
        ## if you want to include contact cost uncomment next 2 lines
            # contact_cost = self.env.contact_cost
            # reward = reward - contact_cost
        if self.task == 'upright':
            reward = reward + 0.1*self._upright(new_obs)
        elif self.task == 'orig':
            pass
        # 5. see if done
        if self.no_done:
            done = False
        elif self.mod_done:
            done = self._done(new_obs)

        return new_obs, reward, done, info

    def _get_obs(self):
        position = self.env.sim.data.qpos.flat.copy()
        velocity = self.env.sim.data.qvel.flat.copy()
        if self.include_contacts:
            contact_force = self.env.contact_forces.flat.copy()
            observations = np.concatenate((position, velocity, contact_force))
        else:
            observations = np.concatenate((position, velocity))
        return observations

    def reset(self):
        self._upside_down_start_time = None
        state = self.env.reset()
        self._update_contacts()
        new_obs = self._get_obs()
        return new_obs
