import os

import neat
import os
import pickle

import neat
import numpy as np
from envs import SnakeEnv1D


def eval_genomes(_genomes, config):
    net = []
    genomes = []
    games = []

    for j, (i, genome) in enumerate(_genomes):
        genome.fitness = 0
        n = neat.nn.FeedForwardNetwork.create(genome, config)
        net.append(n)
        genomes.append(genome)

        game = SnakeEnv1D()
        games.append(game)

    for i, game in enumerate(games):
        obs = game.reset()
        done = False
        while not done:
            action = np.argmax(net[i].activate(obs))
            obs, rewards, done, info = game.step(action)
            genomes[i].fitness += rewards - abs(int(obs[3]))/1000 - abs(int(obs[4]))/1000
        game.close()


def run(config_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    p = neat.Population(config)
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.Checkpointer(1_000_000))
     
    p.run(eval_genomes, 5_000_000)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat-config")
    run(config_path)
