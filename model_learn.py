import os
import time

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from snake_env import SnakeEnv, SnakeEnv1D

models_dir = f'models/{int(time.time())}'
log_dir = f'logs/{int(time.time())}'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

N_ENVS = 4

# env = make_vec_env(SnakeEnv1D, n_envs=N_ENVS, seed=0)
# env = VecFrameStack(env, n_stack=4, channels_order='last')
env = SnakeEnv1D()
env.reset()

model = PPO('MlpPolicy', env, verbose=1, n_steps=5*N_ENVS, tensorboard_log=log_dir)
# model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, learning_starts=100000, exploration_fraction=.0001)

TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS,
                reset_num_timesteps=False, tb_log_name=f"PPO-1D-1envs-no_frames")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
