import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


class SampleEnv:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.grid = np.zeros((n, m), dtype=int)
        # turn 5 squares into bad plants by random
        # save these squares in a list called bad_plant_positions
        self.bad_plant_positions = []
        for i in range(7):
            x = np.random.randint(0, m)
            y = np.random.randint(0, n)
            self.bad_plant_positions.append((x, y))

    def plant(self, x, y):
        self.grid[x, y] = 1
        return self.grid

    def bad_plant(self, x, y):
        self.grid[x, y] = 5
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
        #grid_copy = self.grid.copy()
        grid_copy = self.grid.copy()
        #loop through all spots in copy_grid and call the get_best_action_for_spot method
        #for i in range(len(grid_copy)):
        #    for j in range(len(grid_copy[i])):
       #         grid_copy[i][j] = self.get_best_action_for_spot((i, j), qTable)


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

        # Mark bad plant positions
        #for row, col in self.bad_plant_positions:
            # print out the row and column of the bad plant positions
         #   if grid_copy[row, col] == 1:  # Tree planted in a bad spot
          #      grid_copy[row, col] = 3  # Mark it as orange (3)
           # elif grid_copy[row, col] == -1:  # Bad spot but not planted
            #    grid_copy[row, col] = 2  # Mark it as red (2)

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
        Resets the grid to the initial state with no trees.
        """
        self.grid = np.zeros((self.m, self.n), dtype=int)

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

    def get_possible_actions(self):
        """
        Get all possible actions (positions) in the grid where a tree can be planted.
        """
        return [(i, j) for i in range(self.m) for j in range(self.n) if self.grid[i, j] == 0]

    def get_best_action_for_spot(self, spot, q_table):
        # Initialize the maximum Q-value to a very small number
        max_q_value = -float('inf')
        # Initialize the best action to None
        best_action = None
        # Iterate over all the actions in the Q-table
        for state_action, q_value in q_table.items():
            state, action = state_action
            # If the action corresponds to the spot
            if action[0] == spot:
                # Update the maximum Q-value and the best action if necessary
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_action = action[1]
                    if best_action == None: # No action
                        best_action = 1
        # Return the best action
        #print("Best action for spot", spot, "is", best_action, "with Q-value", max_q_value)
        return best_action


if __name__ == "__main__":
    env = SampleEnv(10, 10)
    tree_positions = [(2, 3), (4, 5), (6, 7), (8, 2), (9, 9)]
    # loop through the tree_positions and plant the trees
    for tree in tree_positions:
        env.plant(tree[0], tree[1])
    env.plot_grid()
