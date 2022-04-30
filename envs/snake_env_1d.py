import random
import time
from collections import deque

import cv2
import gym
import numpy as np
from gym import spaces
from .utils import collision_with_apple, collision_with_boundaries, collision_with_self

# This is used to play trained models back because you need the observation space to be right.
# This model also uses a 1D observation space and not the image of the board.
class SnakeEnv1D(gym.Env):
    def __init__(self, grid_x=10, grid_y=10):
        super(SnakeEnv1D, self).__init__()
        self.SNAKE_LEN_GOAL = int(10*10*.10)
        max_distance = (int(np.sqrt(grid_x**2 + grid_y**2)) + 1)*10
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-max_distance, high=max_distance, shape=(self.SNAKE_LEN_GOAL + 5, ), dtype=np.float32)
        self.grid_x = grid_x
        self.grid_y = grid_y

    def step(self, action):
        button_direction = action
        self.prev_actions.appendleft(button_direction)

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

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        observation = [head_x, head_y, apple_delta_x, apple_delta_y, 
                       snake_length] + list(self.prev_actions)
        observation = np.array(observation, dtype=np.float32)
        return observation, self.reward, self.done, info

    def reset(self):
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

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[0] - head_y

        self.prev_actions = deque(maxlen=self.SNAKE_LEN_GOAL)
        for i in range(self.SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)

        observation = [head_x, head_y, apple_delta_x, 
                       apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation, dtype=np.float32)
        return observation

    def render(self, *args, **kwargs):
        image = np.zeros((self.grid_x * 10, self.grid_y * 10, 3), dtype="uint8")
        cv2.rectangle(image, (self.apple_position[0], self.apple_position[1]),
                      (self.apple_position[0] + 10, self.apple_position[1] + 10),
                      (0, 0, 255), 3)

        for position in self.snake_position:
            cv2.rectangle(image, (position[0], position[1]), 
                          (position[0] + 10, position[1] + 10), (0, 255, 0), 3)
        
        image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA).reshape((500, 500, 3))
        cv2.imshow("a", image)
        cv2.waitKey(1)
