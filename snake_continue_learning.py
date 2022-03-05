from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from snake_env import SnakeEnv

TIME = '1646506651'
START_TIMESTEP = 20000

models_dir = f'models/{TIME}'
model_path = f'{models_dir}/{START_TIMESTEP}.zip'
log_dir = f'logs/{TIME}'

env = make_vec_env(SnakeEnv, n_envs=4)
env = VecFrameStack(env, n_stack=4)
env.reset()

model = PPO.load(model_path, env=env, tensorboard_log=log_dir)

TIMESTEPS = 10000
iters = START_TIMESTEP//10000
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS,
                reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")