import random
import time
from collections import deque

import cv2
import gym
import numpy as np
from gym import spaces

SNAKE_LEN_GOAL = 30


def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1, 50)*10, random.randrange(1, 50)*10]
    score += 1
    return apple_position, score


def collision_with_boundaries(snake_head):
    if snake_head[0] >= 500 or snake_head[0] < 0 or snake_head[1] >= 500 or snake_head[1] < 0:
        return 1
    else:
        return 0


def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0


class SnakeEnv(gym.Env):
    def __init__(self):
        super(SnakeEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(50, 50, 1), dtype='uint8')

    def step(self, action):
        button_direction = action

        # Change the head position based on the button direction
        if button_direction == 1:
            self.snake_head[0] += 10
        elif button_direction == 0:
            self.snake_head[0] -= 10
        elif button_direction == 2:
            self.snake_head[1] += 10
        elif button_direction == 3:
            self.snake_head[1] -= 10

        apple_reward = 0
        self.time_without_apple += 1
        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0, list(self.snake_head))
            self.grid[self.snake_head[0]//10-1, self.snake_head[1]//10-1, 0] = 125
            self.grid[self.apple_position[0]//10-1, self.apple_position[1]//10-1, 0] = 255
            apple_reward = 1
            self.time_without_apple = 0
        else:
            self.grid[self.snake_head[0]//10-1, self.snake_head[1]//10-1, 0] = 125
            self.grid[self.snake_position[-1][0]//10-1, self.snake_position[-1][1]//10-1, 0] = 0
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()
            self.time_without_apple += 1

        death_reward = 0
        # On collision kill the snake and print the score
        if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1 or self.time_without_apple > 300:
            self.done = True
            death_reward = -1

        self.reward = apple_reward + death_reward

        info = {}
        observation = self.grid

        return observation, self.reward, self.done, info

    def reset(self):
        self.img = np.zeros((500, 500, 3), dtype='uint8')
        # Initial Snake and Apple position
        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        self.apple_position = [random.randrange(1, 50)*10, random.randrange(1, 50)*10]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [250, 250]

        self.time_without_apple = 0
        self.done = False

        self.grid = np.zeros((50, 50, 1), dtype='uint8')
        self.grid[self.apple_position[0]//10-1, self.apple_position[1]//10-1, 0] = 255
        for position in self.snake_position:
            self.grid[position[0]//10-1, position[1]//10-1, 0] = 125

        observation = self.grid

        return observation

    def render(self, mode=None):
        cv2.imshow('a', self.img)
        cv2.waitKey(1)
        self.img = np.zeros((500, 500, 3), dtype='uint8')
        # Display Apple
        cv2.rectangle(self.img, (self.apple_position[0], self.apple_position[1]), (self.apple_position[0]+10, self.apple_position[1]+10), (0, 0, 255), 3)
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(self.img, (position[0], position[1]), (position[0]+10, position[1]+10), (0, 255, 0), 3)


        if mode == "human":
            # Takes step after fixed time
            t_end = time.time() + 0.05
            k = -1
            while time.time() < t_end:
                if k == -1:
                    k = cv2.waitKey(1)
                else:
                    continue


# This is used to play trained models back because you need the observation space to be right.
# This model also uses a 1D observation space and not the image of the board.
class SnakeEnv1D(gym.Env):
    def __init__(self):
        super(SnakeEnv1D, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-500, high=500, shape=(SNAKE_LEN_GOAL+5, ), dtype=np.float32)

    def step(self, action):
        button_direction = action

        # Change the head position based on the button direction
        if button_direction == 1:
            self.snake_head[0] += 10
        elif button_direction == 0:
            self.snake_head[0] -= 10
        elif button_direction == 2:
            self.snake_head[1] += 10
        elif button_direction == 3:
            self.snake_head[1] -= 10

        apple_reward = 0
        self.time_without_apple += 1
        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0, list(self.snake_head))
            apple_reward = 1
            self.time_without_apple = 0
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()
            self.time_without_apple += 1

        death_reward = 0
        # On collision kill the snake and print the score
        if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1 or self.time_without_apple > 300:
            self.done = True
            death_reward = -1

        self.reward = apple_reward + death_reward

        info = {}

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation)

        return observation, self.reward, self.done, info

    def reset(self):
        self.img = np.zeros((500, 500, 3), dtype='uint8')
        # Initial Snake and Apple position
        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        self.apple_position = [random.randrange(
            1, 50)*10, random.randrange(1, 50)*10]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [250, 250]

        self.time_without_apple = 0
        self.done = False

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[0] - head_y

        self.prev_actions = deque(maxlen=SNAKE_LEN_GOAL)
        for i in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)

        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation)

        return observation

    def render(self, mode=None):
        cv2.imshow('a', self.img)
        cv2.waitKey(1)
        self.img = np.zeros((500, 500, 3), dtype='uint8')
        # Display Apple
        cv2.rectangle(self.img, (self.apple_position[0], self.apple_position[1]), (self.apple_position[0]+10, self.apple_position[1]+10), (0, 0, 255), 3)
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(self.img, (position[0], position[1]), (position[0]+10, position[1]+10), (0, 255, 0), 3)


        if mode == "human":
            # Takes step after fixed time
            t_end = time.time() + 0.05
            k = -1
            while time.time() < t_end:
                if k == -1:
                    k = cv2.waitKey(1)
                else:
                    continue
