import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from landscape.Square import Square
class Grid:

    def __init__(self, x, y):
        """
        Constructor for the Grid class
        """
        self.x = x
        self.y = y
        self.landscapeArea = x * y
        self.grid = self.createGrid(x, y)



    def createGrid(self, x, y):
        """
        Method to create a grid of squares
        :param x: x size of the grid
        :param y: y size of the grid
        :return: the grid
        """
        self.grid = np.empty((x, y), dtype=object)

        for i in range(x):
            for j in range(y):
                if i == 7 and j < 4:
                    self.grid[i, j] = Square(i, j, True, True)
                elif i > 4 and j > 3:
                    self.grid[i, j] = Square(i, j, False, False)
                    self.landscapeArea-=1
                else:
                    self.grid[i, j] = Square(i, j, False, True)
        return self.grid

    def plant(self, x, y, tree):
        self.grid[x, y].plant(tree)
        return self.grid


    def plot_grid(self, qTable):
        """
        Plots the grid using Matplotlib.
        - Good planting spots (where trees are planted) are green.
        - Bad planting spots (whether planted or not) are red.
        - Bad planting spots (where trees are planted) are orange.
        - Unplanted spots remain white.
        """
        # Create a copy of the grid for plotting
        grid_copy = self.grid.copy()

        for i in range(len(grid_copy)):
            for j in range(len(grid_copy[i])):
                action = self.get_best_action_for_spot((i, j), qTable)
                print(action, end=" ")
                if action == 1 and (i, j) in self.bad_plant_positions:  # Tree planted in a bad spot
                    grid_copy[i][j] = 3  # Mark it as orange (3)
                elif action == 1:  # Good spot with a tree
                    grid_copy[i][j] = 1  # Mark it as green (1)
                elif action == -1 and (i, j) in self.bad_plant_positions:  # Bad spot but not planted
                    grid_copy[i][j] = 2  # Mark it as red (2)
                else:  # Unplanted spot
                    grid_copy[i][j] = 0  # Mark it as white (0)
            print()

        print("--------------------")

        #make the grid_copy a numpy array
        grid_copy = np.array(grid_copy)
        #print out the values in grid_copy
        for i in range(len(grid_copy)):
            for j in range(len(grid_copy[i])):
                print(grid_copy[i][j], end=" ")
            print()
        # Define a colormap with four colors: white, green, red, orange
        cmap = colors.ListedColormap(['green', 'red', 'orange'])

        plt.figure(figsize=(8, 8))
        plt.imshow(grid_copy, cmap=cmap, interpolation="none")
        plt.grid(True, which='both', color='black', linestyle='-', linewidth=2)
        plt.xticks(np.arange(-0.5, self.grid.shape[1], 1), [])
        plt.yticks(np.arange(-0.5, self.grid.shape[0], 1), [])
        plt.gca().set_xticks(np.arange(-0.5, self.grid.shape[1], 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, self.grid.shape[0], 1), minor=True)
        plt.gca().grid(which='minor', color='black', linestyle='-', linewidth=2)
        #plt.gca().invert_yaxis()
        plt.savefig('grid.png')  # Save the plot as an image
        plt.show(block=True)  # Keep the plot open to inspect

    def reset(self):
        """
        Method to reset the grid
        :return: the grid
        """
        self.grid = np.zeros((self.x, self.y), dtype=int)
        return self.grid

    def step(self, action):
        """
        Method to take a step in the environment
        :param action: the action to take
        :return: the new state, reward, and done
        """
        pass

    def get_possible_actions(self):
        """
        Get all possible actions (positions) in the grid where a tree can be planted.
        """
        return [(i, j) for i in range(self.x) for j in range(self.y) if self.grid[i, j].plantable]

    def step(self, action):
        """
        Perform an action in the environment.

        Parameters:
        - action: a tuple (row, col) indicating where to plant a tree, or None for no action.

        Returns:
        - state: the new state after performing the action.
        - reward: the reward received after performing the action.
        - done: a boolean indicating if the episode is done.
        """
        reward = 0
        if action[1] is not None:
            row, col = action[0]
            if (row, col) in self.bad_plant_positions:
                reward = 3
            else:
                reward = -1
            self.grid[row, col] = -1
            state = self.grid.copy()
            done = np.all(self.grid != 0)
            return state, reward, done

        row, col = action[0]

        if self.grid[row, col] == 0:  # Only act if the spot is empty
            if (row, col) in self.bad_plant_positions:
                reward = -3  # Punishment for planting in a bad spot
            else:
                reward = 1  # Reward for planting in a good spot
            self.grid[row, col] = 1  # Mark the spot as planted

        state = self.grid.copy()  # The new state is the updated grid
        done = np.all(self.grid != 0)  # Episode ends if all cells are filled
        return state, reward, done