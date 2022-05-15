import random

import cv2
import gym
import numpy as np
from gym import spaces

from .utils import collision_with_apple, collision_with_boundaries, collision_with_self


class SnakeEnv(gym.Env):
    def __init__(self, grid_x=10, grid_y=10):
        super(SnakeEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(100, 100, 1), dtype="uint8")
        self.grid_x = grid_x
        self.grid_y = grid_y

    def _get_observation(self):
        self.img = np.zeros((self.grid_x * 10, self.grid_y * 10, 3), dtype="uint8")
        cv2.rectangle(self.img, (self.apple_position[0], self.apple_position[1]),
                      (self.apple_position[0] + 10, self.apple_position[1] + 10), (0, 0, 255), 3)

        for position in self.snake_position:
            cv2.rectangle(self.img, (position[0], position[1]), (position[0] + 10, position[1] + 10),
                          (0, 255, 0), 3)

        return cv2.resize(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), 
                          (100, 100), interpolation=cv2.INTER_AREA).reshape((100, 100, 1))

    def _get_render(self):
        render = np.zeros((self.grid_x * 10, self.grid_y * 10, 3), dtype="uint8")
        cv2.rectangle(render, (self.apple_position[0], self.apple_position[1]),
                      (self.apple_position[0] + 10, self.apple_position[1] + 10), (0, 0, 255), 3)

        for position in self.snake_position:
            cv2.rectangle(render, (position[0], position[1]), (position[0] + 10, position[1] + 10),
                          (0, 255, 0), 3)

        return cv2.resize(render, (500, 500), interpolation=cv2.INTER_AREA).reshape((500, 500, 3))

    def step(self, action):
        button_direction = action

        if button_direction == 1:
            self.snake_head[0] += 10
        elif button_direction == 0:
            self.snake_head[0] -= 10
        elif button_direction == 2:
            self.snake_head[1] += 10
        elif button_direction == 3:
            self.snake_head[1] -= 10

        apple_reward = 0

        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score, self.grid_x, self.grid_y)
            self.snake_position.insert(0, list(self.snake_head))
            apple_reward = 1
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        death_reward = 0
        if (collision_with_boundaries(self.snake_head, self.grid_x, self.grid_y) == 1
            or collision_with_self(self.snake_position) == 1):
            self.done = True
            death_reward = -1

        self.reward = apple_reward + death_reward

        info = {}
        observation = self._get_observation()

        return observation, self.reward, self.done, info

    def reset(self):
        self.img = np.zeros((self.grid_x * 10, self.grid_y * 10, 3), dtype="uint8")

        self.snake_position = [[self.grid_x // 2 * 10, self.grid_y // 2 * 10],
                               [(self.grid_x // 2 - 1) * 10, self.grid_y // 2 * 10],
                               [(self.grid_x // 2 - 2) * 10, self.grid_y * 10 // 2]]
        self.apple_position = [random.randrange(1, self.grid_x) * 10,
                               random.randrange(1, self.grid_y) * 10]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [self.grid_x // 2 * 10, self.grid_y // 2 * 10]

        self.done = False

        observation = self._get_observation()
        return observation

    def render(self, *args, **kwargs):
        image = self._get_render()
        cv2.imshow("a", image)
        cv2.waitKey(1)
