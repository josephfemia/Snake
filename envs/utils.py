import random


def collision_with_apple(apple_position, score, GRID_X=50, GRID_Y=50):
    apple_position = [random.randrange(1, GRID_X) * 10,
                      random.randrange(1, GRID_Y) * 10]
    score += 1
    return apple_position, score


def collision_with_boundaries(snake_head, GRID_X=50, GRID_Y=50):
    if (snake_head[0] >= GRID_X * 10 or snake_head[0] < 0
        or snake_head[1] >= GRID_Y * 10 or snake_head[1] < 0):
        return 1
    else:
        return 0


def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0
