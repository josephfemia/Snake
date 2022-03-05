import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from snake_env import SnakeEnv

models_dir = f'models/{int(time.time())}'
log_dir = f'logs/{int(time.time())}'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = make_vec_env(SnakeEnv, n_envs=4)
env = VecFrameStack(env, n_stack=4)
env.reset()

model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS,
                reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
