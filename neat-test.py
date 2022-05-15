import os
import pickle

import neat

from envs import SnakeEnv1D

with open(os.path.join("models", "neat", "best.pkl"), "rb") as f:
    best = pickle.load(f)


print("Loaded genome:")
print(best)

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, "neat-config")
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path,
)

net = neat.nn.FeedForwardNetwork.create(best, config)
env = SnakeEnv1D(grid_x=10, grid_y=10)
observation = env.reset()
done = False

while not done:
    action = net.activate(observation)
    observation, reward, done, info = env.step(action)
    env.render()
