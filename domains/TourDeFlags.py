'''
Modified on Sep. 22, 2018

@author: Michael Gimelfarb
'''
import numpy as np
from matplotlib import pyplot as plt
from domains.Domain import Domain

# possible movement directions
LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3


class TourDeFlags(Domain):

    def __init__(self, maze, initial):
        super().__init__()
        self.initial = (initial[0], initial[1], 0)
        self.maze = maze
        self.height, self.width = maze.shape

        # count all the sub-goals and free cells
        self.goals = 0
        self.free_cells = set()
        for c in range(self.width):
            for r in range(self.height):
                if maze[r, c] > 0:
                    self.goals = max(self.goals, maze[r, c])
                elif maze[r, c] == 0:
                    self.free_cells.add((r, c))

        # max steps to the goal
        self.max_steps = max(200, self.goals * (self.width + self.height))

    def copy(self):
        return TourDeFlags(np.copy(self.maze), self.initial)

    def initial_state(self):
        return self.initial

    def valid_actions(self):
        return [LEFT, UP, RIGHT, DOWN]

    def check(self, row, col, action):
        if (row == 0 and action == UP) or \
            (row == self.height - 1 and action == DOWN) or \
            (col == 0 and action == LEFT) or \
            (col == self.width - 1 and action == RIGHT):
            return False
        return True

    def act(self, state, action):

        # perform the movement
        row, col, collected = state
        if self.check(row, col, action):
            if action == LEFT:
                col -= 1
            elif action == UP:
                row -= 1
            elif action == RIGHT:
                col += 1
            elif action == DOWN:
                row += 1
        else:
            return (row, col, collected), -2.0, False

        # compute the new state, status and reward
        grid_x1y1 = self.maze[row, col]
        if grid_x1y1 > 0:
            if collected + 1 == grid_x1y1:
                if collected + 1 == self.goals:
                    return (row, col, collected + 1), -1.0, True
                else:
                    return (row, col, collected + 1), -1.0, False
            elif collected >= grid_x1y1:
                return (row, col, collected), -1.0, False
            else:
                return (row, col, collected), -2.0, False
        else:
            return (row, col, collected), -1.0, False

    def render(self, policy, encoder, time=0.02):
        act = self.act
        state = self.initial_state()
        done = False
        im = None
        grid = np.copy(self.maze).astype('float')
        for row in range(self.height):
            for col in range(self.width):
                if grid[row, col] == -1:
                    grid[row, col] = 0.0
                elif grid[row, col] == 0:
                    grid[row, col] = 1.0
                else:
                    grid[row, col] = 0.5
        steps = 0
        while not done and steps < self.max_steps:
            state_enc = encoder(state)
            action = policy(state_enc)
            new_state, _, done = act(state, action)
            if state == new_state:
                break
            row, col, _ = state = new_state
            grid[row, col] = 0.25
            if not im:
                im = plt.imshow(grid, cmap='gray', vmin=0, vmax=1)
            else:
                im.set_data(grid)
            plt.draw()
            plt.pause(time)
            grid[row, col] = 1.0
            steps += 1

    def default_encoding(self, state):

        # encoding is [e(row), e(col), e(collected)]^T
        input_dim = self.width + self.height + (1 + self.goals)
        arr = np.zeros(input_dim)
        row, col, collected = state
        arr[row] = 1.0
        arr[self.height + col] = 1.0
        arr[self.height + self.width + collected] = 1.0
        arr = arr.reshape((1, -1))
        return arr
