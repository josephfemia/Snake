from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
import os
import time
from envs import SnakeEnv, SnakeEnv1D

model_path = os.path.join("models", "full", "1650815114", "model_5000000.zip")

# env = SnakeEnv()
env = make_vec_env(SnakeEnv, n_envs=1, env_kwargs={"grid_x": 10, "grid_y": 10})
env = VecFrameStack(env, n_stack=4, channels_order="last")

model = PPO.load(model_path, env=env)

for _ in range(20):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        # time.sleep(0.01)
    env.close()
