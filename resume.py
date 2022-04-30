from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from callbacks.TrainAndLoggingCallback import TrainAndLoggingCallback
import os
from envs import SnakeEnv, SnakeEnv1D

TIME = "1650818573"
START_TIMESTEP = 5000000

models_dir = os.path.join("models", "full", f"{TIME}")
log_dir = os.path.join("logs", "full", f"{TIME}")

N_ENVS = 4

model_params = {
    "n_steps": 7488,
    "gamma": 0.7667437823139281,
    "learning_rate": 7.788317778454502e-05,
    "clip_range": 0.3,
    "gae_lambda": 0.8860384931245284,
}

env = make_vec_env(SnakeEnv, n_envs=N_ENVS, env_kwargs={"grid_x": 10, "grid_y": 10})
env = VecFrameStack(env, n_stack=4, channels_order="last")
env.reset()

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir, **model_params)
model.set_parameters(os.path.join("models", "full", f"{TIME}", "model_1250000.zip"))

# model.learn(total_timesteps=5_000_000,
#             callback=TrainAndLoggingCallback(check_freq=max(50_000 // N_ENVS, 1), 
#                                              save_path=models_dir, 
#                                              n_calls_offset=0))

env = make_vec_env(SnakeEnv, n_envs=N_ENVS, env_kwargs={"grid_x": 20, "grid_y": 20})
env = VecFrameStack(env, 4, channels_order="last")
model.set_env(env)
model.learn(total_timesteps=10_000_000,
            reset_num_timesteps=False,
            callback=TrainAndLoggingCallback(check_freq=max(50_000 // N_ENVS, 1), 
                                             save_path=models_dir, 
                                             n_calls_offset=5000000))

env = make_vec_env(SnakeEnv, n_envs=N_ENVS, env_kwargs={"grid_x": 30, "grid_y": 30})
env = VecFrameStack(env, 4, channels_order="last")
model.set_env(env)
model.learn(total_timesteps=15_000_000,
            reset_num_timesteps=False,
            callback=TrainAndLoggingCallback(check_freq=max(50_000 // N_ENVS, 1), 
                                             save_path=models_dir, 
                                             n_calls_offset=15000000))

env = make_vec_env(SnakeEnv, n_envs=N_ENVS, env_kwargs={"grid_x": 40, "grid_y": 40})
env = VecFrameStack(env, 4, channels_order="last")
model.set_env(env)
model.learn(total_timesteps=20_000_000,
            reset_num_timesteps=False,
            callback=TrainAndLoggingCallback(check_freq=max(50_000 // N_ENVS, 1), 
                                             save_path=models_dir, 
                                             n_calls_offset=30000000))

env = make_vec_env(SnakeEnv, n_envs=N_ENVS, env_kwargs={"grid_x": 50, "grid_y": 50})
env = VecFrameStack(env, 4, channels_order="last")
model.set_env(env)
model.learn(total_timesteps=25_000_000,
            reset_num_timesteps=False,
            callback=TrainAndLoggingCallback(check_freq=max(50_000 // N_ENVS, 1), 
                                             save_path=models_dir, 
                                             n_calls_offset=50000000))
