import os
import time

import cv2
import numpy as np
from envs import SnakeEnv, SnakeEnv1D
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

env = SnakeEnv1D()
check_env(env)

# More custom testing code if more in-detail tetsing needs to take place
env = make_vec_env(SnakeEnv1D, n_envs=1, env_kwargs={'grid_x': 10, 'grid_y': 10})
# env = VecFrameStack(env, n_stack=3, channels_order='last')

for _ in range(5):
    obs = env.reset()
    done = False
    while not done:
        action, _states = [2, []]
        obs, rewards, done, info = env.step([action])
        env.render()
        time.sleep(0.1)
    env.close()
