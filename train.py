import os
import time

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

from envs import SnakeEnv, SnakeEnv1D
from callbacks.TrainAndLoggingCallback import TrainAndLoggingCallback

N_ENVS = 8
grid_x = 20
grid_y = 20
obs_dims = "2D"

model_params_10x10 = {
    "n_steps": 7488,
    "gamma": 0.7667437823139281,
    "learning_rate": 7.788317778454502e-05,
    "clip_range": 0.3,
    "gae_lambda": 0.8860384931245284,
}

model_params_20x20 = {
    "n_steps": 7488,
    "gamma": 0.7667437823139281,
    "learning_rate": 7.788317778454502e-05,
    "clip_range": 0.3,
    "gae_lambda": 0.8860384931245284,
}

models_dir = os.path.join("models", obs_dims, f"{grid_x}x{grid_y}",  f"{int(time.time())}")
log_dir = os.path.join("logs", obs_dims, f"{grid_x}x{grid_y}", f"{int(time.time())}")

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = make_vec_env(SnakeEnv, n_envs=N_ENVS, env_kwargs={"grid_x": grid_x, "grid_y": grid_y})
env = VecFrameStack(env, n_stack=4, channels_order="last")
env.reset()

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir, **model_params_20x20)
model.set_parameters(os.path.join("opt", "models", obs_dims, f"{grid_x}x{grid_y}", "trial_x.zip"))

model.learn(total_timesteps=10_000_000,
            callback=TrainAndLoggingCallback(check_freq=max(50_000 // N_ENVS, 1), 
                                             save_path=models_dir, 
                                             n_calls_offset=0))
