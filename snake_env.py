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


"""
Model Notes:

1644687068, PPO
- Reward for getting an apple, 100,000
- Reward for losing the game, -10
- Reward functions code:

    euclidean_dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))
    self.total_reward = ((250 - euclidean_dist_to_apple) + apple_reward) / 100
    self.reward = self.total_reward - self.prev_reward
    self.prev_reward = self.total_reward

- Observations code:
    head_x = self.snake_head[0]
    head_y = self.snake_head[1]

    snake_length = len(self.snake_position)
    apple_delta_x = self.apple_position[0] - head_x
    apple_delta_y = self.apple_position[1] - head_y

    observation = [head_x, head_y, apple_delta_x,
        apple_delta_y, snake_length] + list(self.prev_actions)

- For exact code check commit: 31ac5e3b26478aca24dc50da4e6de27c35336e28 (the only difference will be the reward for getting an apple)

1644690486, PPO
- Same thing as 1644687068, but diff reward for getting an apple 
- Reward for getting an apple, 10,000

1644690553, PPO
- Same thing as 1644687068, but diff reward for getting an apple 
- Reward for getting an apple, 1,000

1644701321 and 1644733599, PPO
- Reward for getting an apple, 25
- 1/x function that rewards more and more the closer to apple (exp on euclidean dist)
- Reward function tuned to use a 1/x function rather than something linear:
     euclidean_dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))
    self.total_reward = (450/(euclidean_dist_to_apple+1)) + apple_reward
    self.reward = self.total_reward - self.prev_reward
    self.prev_reward = self.total_reward
- The rest is the same as 1644687068, commit hash: bf764cc60296c264169e3c081fe4fb15be722ec7

1644846992, PPO
- Reward for getting an apple, 100
- Reward for dying is decreased to -1000
- The rest is the same as 1644701321 and 1644733599, commit hash: 3ea8ff1ded699d051abf9225ef540f65e23df1f5

1644853531, PPO
- Reward for dying is decreased to -100
- The rest is the same as 1644846992

1644873612, PPO
- Reward for getting an apple increased to 1000
- The rest is the same as 1644846992

1645886845, PPO
- Euclidian Distance moved to the observation space
- Apple reward = 10
- Dying reward = -10
- Reward of -.05 for taking each step has now been implemented
- Pretty major change, commit hash: 288cb78e064f0b91527802040f07b0ca0e335b69

1645887534, PPO
- Apple reward decreased to 1
- Reward for dying increased to -1
- The rest is the same as 1645886845

1645888751, PPO
- Removed Euclidian Distance from the observation space
- Apple reward increased to 5
- Reward for dying decreased to -5
- Changed total reward to include euclidian distance to:
    self.total_reward = (-1.5**(-euclidean_dist_to_apple/700)) - .05 + apple_reward 
- The rest is the same as 1645886845, commit hash: 8eba3ba98779e51aba89a57033fc9d3f08ce4a33

1645894485, PPO
- Removed comparison of total and previous rewards
- Removed issue where overwrite reward with a constant when dead.
- The rest is the same as 1645888751, commit hash: 66d176adc7696d7e66a0c4b2faaabf92a783e08f

1645896013, PPO
- Changed reward to include euclidian distance to:
    self.reward = (1.5**(-euclidean_dist_to_apple/700)) + apple_reward + death_reward
- The rest is the same as 1645894485

1645896654, PPO
- Updated reward function:
    self.reward = 2.5 - (2.5*euclidean_dist_to_apple/700) + apple_reward + death_reward
- The rest is the same as 1645894485

1645904983, PPO
- Apple reward increased to 50
- Death reward decreased to -25
- Updated reward function:
    self.reward = -(30*euclidean_dist_to_apple/700) + apple_reward + death_reward
- The rest is the same as 1645894485

1645905867, PPO
- Add an episode_length veriable to track how long the snake has been going for
- Apple reward of 50
- Death reward of -1 * self.episode_length * len(self.snake_position)
- Trigger the game to end in 1000 steps
- Changed reward function to:
    self.reward = -euclidean_dist_to_apple/10 + apple_reward + death_reward
- The rest is the same as 1645894485, commit hash: b50939b3eb7634a1203f226125ba7a5920d28457
"""


class SnakeEnv(gym.Env):
    def __init__(self):
        super(SnakeEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input (channel-first; channel-last also works):
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
        self.episode_length += 1
        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0, list(self.snake_head))
            apple_reward = 50

        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        death_reward = 0
        # On collision kill the snake and print the score
        if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1 or self.episode_length == 1000:
            self.done = True
            death_reward = -1 * self.episode_length * len(self.snake_position)

        euclidean_dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))
        self.reward = -euclidean_dist_to_apple/10 + apple_reward + death_reward

        info = {}

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        # create observation:
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

        self.episode_length = 0
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
