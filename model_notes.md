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

1645907877, PPO - Best model so far (28610000.zip)
- Add variable to make snake die if goes certain amount of steps without an apple
- Apple reward = 1
- Death reward = -1
- Remove euclidian distance (no longer involved in observation space or reward structure)
- Updated reward function:
    self.reward = apple_reward + death_reward
- The rest is the same as 1645894485, commit hash: 028fbd92684474c2c32f42635d2dd86a937f8014

1645915125, PPO
- Increase time_without_apple to 1000 steps
- Add previous action population
- Store up to 30 coordiantes for the snake body positions, constantly updated with snake moving
    - The reason for this list having a len of 60 is because 30 body parts * 2 (for each of the x and y coordinates of the body part)
- The rest is the same as 1645907877, commit hash: 8033e77393a7a1f891a90f9e42609eb661bb88fe

1645916294, PPO
- Decrease time_without_apple to 500 steps
- The rest is the same as 1645915125

1645916776, PPO
- Remove previous actions population
- The rest is the same as 1645915125

1646486111, PPO
- Pass in image of board as observation space
- Remove all other previous stuff used in observation space
- Same time_without_apple as 1645916294
- Use CnnPolicy for learning
- Major change to environment, commit hash: 2362f13a406d22a4431d9367c7aac767fa6b7c8c

1646488273, PPO
- Speed up grid population
- The rest is the same as 1646486111

1646490862, PPO
- Vectorizing the environment with 4 environments
- The rest is the same as 1646488273

Note: 1646486111, 1646488273, and 1646490862 are not trained too much because of 500x500 grid takes forever to train.
Also the game was really a 50x50 game with and additional 10 dimensions for drawing so we can actually see the snake when visualizing

1646506651, PPO
- Found a bug where we were over estimating the board size by 10
- The reason for this is that we need to be able to draw the snake, but for our grid for RL each snake position is 1x1 not 10x10
- This significantly decreases training times
- Uses 4 Vectorized envs with a frame stack of 4
- Pretty big fix in 2D environment, commit hash: 33adfd38b9a4b254157bee849b5beef7e75d9100

1646507207, PPO
- Increases the amount of vectorized envs to 10, and 10 frame stacks
- The rest is the same as 1646506651

1646509458, PPO
- Use an RGB matrix as grid
- The rest is the same as 1646507207

1646510903, PPO
- Uses 10 Vectorized envs with a frame stack of 4
- Decreased time_without_apple down to 300
- The rest is the same as 1646506651

1646515016, A2C
- Used A2C model
- The rest is the same as 1646510903

1646516997, PPO
- Fixes some grid population bugs
- Commit hash: 2db2d274f37d074496434464f22ff06f0ae4d95b

1646519447, PPO
- 4 vectorized env, with a 4 frame stack
- TIMESTEPS = 10000*4
- The rest is the same as 1646516997

1646519789, PPO
- TIMESTEPS = 10000*4*4
- The rest is the same as 1646519447

1646520469, PPO
- Just 1 environment with no frame stacking
- The rest is the same as 1646516997

1646524486, PPO
- 16 vectorized env, with a 4 frame stack
- TIMESTEPS = 5_000_000
- The rest is the same as 1646516997


Finally got proper 2D Snake observations working
- Finally made 2D snake observation space work
- Major fix to obervation space generation
- cv2 switches x and y axis when drawing
- Support multiple different board sizes for 2D env
- Also found and fixed something weird when looking at rendered env, it was always 1 frame behind
- Commit hash: f6a41fe4d9a3cfcd31a83ebca5fa89b012ed2646
