import os
import time

from stable_baselines3 import PPO

from snake_env import SnakeEnv

models_dir = f'models/{int(time.time())}'
log_dir = f'logs/{int(time.time())}'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = SnakeEnv()
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS,
                reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
