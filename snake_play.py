from stable_baselines3 import PPO

from snake_env import SnakeEnv, SnakeEnv1D

models_dir = f'models/1645980703'
model_path = f'{models_dir}/10000.zip'

env = SnakeEnv()
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
