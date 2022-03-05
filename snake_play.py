from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from snake_env import SnakeEnv, SnakeEnv1D

models_dir = f'models/1646506651'
model_path = f'{models_dir}/590000.zip'

# env = SnakeEnv()
env = make_vec_env(SnakeEnv, n_envs=1)
env = VecFrameStack(env, n_stack=4)
env.reset()

model = PPO.load(model_path, env=env)

while True:
    obs = env.reset()
    done = False

    while not done:
        env.render()
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        # print(rewards)
    env.close()
