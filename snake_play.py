from stable_baselines3 import PPO

from snake_play_environment import OGSnakeEnv
from snake_env import SnakeEnv

models_dir = f'models/1645907877'
model_path = f'{models_dir}/28610000.zip'

env = OGSnakeEnv()
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
