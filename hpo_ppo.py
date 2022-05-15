import csv
import os

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env

from envs import SnakeEnv, SnakeEnv1D

grid_x = 50
grid_y = 50
obs_dims = "2D"
load_path = os.path.join("models", obs_dims, "40x40", "1651926724", "model_2137500.zip")

def optimize_ppo(trial):
    return {
        "n_steps": trial.suggest_int("n_steps", 64, 2048, 64),
        "gamma": trial.suggest_uniform("gamma", 0.9, 0.9999),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 1e-4),
        "clip_range": trial.suggest_uniform("clip_range", 0.1, 0.4),
        "gae_lambda": trial.suggest_uniform("gae_lambda", 0.8, 0.99),
    }


def optimize_agent(trial):
    try:
        model_params = optimize_ppo(trial)

        env = make_vec_env(SnakeEnv, n_envs=8, env_kwargs={"grid_x": grid_x, "grid_y": grid_y}, seed=0)
        env = VecFrameStack(env, 4, channels_order="last")

        model = PPO("CnnPolicy", env, tensorboard_log=os.path.join("opt", "logs", obs_dims, f"{grid_x}x{grid_y}"),
                    verbose=0, **model_params)
        model.set_parameters(load_path)
        model.learn(total_timesteps=100_000)
        env.close()
        
        mean_reward = 0

        SAVE_PATH = os.path.join("opt", "models", obs_dims, f"{grid_x}x{grid_y}", f"trial_{trial.number}")
        model.save(SAVE_PATH)

        if trial.number == 0:
            flag = "w"
        else:
            flag = "a"

        with open(os.path.join("opt", "logs", obs_dims, f"{grid_x}x{grid_y}", "summary.csv"), flag, newline="") as f:
            csv_writer = csv.writer(f)
            if trial.number == 0:
                csv_writer.writerow(["Started From", "n_eval_episodes", "total_timesteps",
                                     "n_trials", "Grid Size", "PPO_n_steps", "PPO_gamma",
                                     "PPO_learning_rate", "PPO_clip_range", "PPO_gae_lambda",
                                     "trial_number", "mean_reward"])
            csv_writer.writerow([load_path, "2", "100000", "500", f"{grid_x}x{grid_y}", 
                                 model_params["n_steps"], model_params["gamma"], 
                                 model_params["learning_rate"], model_params["clip_range"], 
                                 model_params["gae_lambda"], trial.number, mean_reward])

        return mean_reward

    except Exception as e:
        print(e)
        return -1000


# study = optuna.create_study(direction="maximize", 
#                             study_name="worms50x50", 
#                             storage="postgresql://postgres:admin@localhost/snakes")
study = optuna.load_study(study_name="worms50x50", 
                          storage="postgresql://postgres:admin@localhost/snakes")
study.optimize(optimize_agent, n_trials=500)
