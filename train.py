import os
import time

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from callbacks.TrainAndLoggingCallback import TrainAndLoggingCallback
from envs import SnakeEnv, SnakeEnv1D
from model_params import model_params

N_ENVS = 8
grid_x = 50
grid_y = 50
obs_dims = "2D"

models_dir = os.path.join("models", obs_dims, f"{grid_x}x{grid_y}",  f"{int(time.time())}")
log_dir = os.path.join("logs", obs_dims, f"{grid_x}x{grid_y}", f"{int(time.time())}")

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = make_vec_env(SnakeEnv, n_envs=N_ENVS, env_kwargs={"grid_x": grid_x, "grid_y": grid_y}, seed=0)
env = VecFrameStack(env, n_stack=4, channels_order="last")
env.reset()

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir, **model_params[f"{grid_x}x{grid_y}"])
model.set_parameters(os.path.join("opt", "models", obs_dims, f"{grid_x}x{grid_y}", "trial_109.zip"))

model.learn(total_timesteps=35_000_000,
            callback=TrainAndLoggingCallback(check_freq=max(50_000 // N_ENVS, 1), 
                                             save_path=models_dir, 
                                             n_calls_offset=0))
