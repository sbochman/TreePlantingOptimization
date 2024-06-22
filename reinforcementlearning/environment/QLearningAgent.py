import numpy as np
import random

class QLearningAgent:
    def __init__(self, environment, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2):
        self.environment = environment
        self.learning_rate = learning_rate  # Alpha
        self.discount_factor = discount_factor  # Gamma
        self.exploration_rate = exploration_rate  # Epsilon
        self.q_table = {}  # Q-table to store Q-values for state-action pairs

    def choose_action(self, state):
        """
        Choose an action based on the epsilon-greedy strategy.
        Include 'no action' as a possible choice.
        """
        possible_actions = self.environment.get_possible_actions()

        if random.uniform(0, 1) < self.exploration_rate:
            # Explore: choose a random action
            if random.uniform(0, 1) < 0.1:
                return random.choice(possible_actions), -1
            return random.choice(possible_actions), None
        else:
            # Exploit: choose the best action from Q-table
            q_values = [self.q_table.get((self.state_to_tuple(state), action), 0) for action in possible_actions]
            max_q_value = max(q_values)
            return random.choice([action for action, q in zip(possible_actions, q_values) if q == max_q_value]), None

    def learn(self, state, action, reward, next_state):
        """
        Update the Q-value for the given state-action pair.
        """
        current_q_value = self.q_table.get((self.state_to_tuple(state), action), 0)
        next_possible_actions = self.environment.get_possible_actions()
        future_q_values = [self.q_table.get((self.state_to_tuple(next_state), next_action), 0) for next_action in next_possible_actions]
        max_future_q_value = max(future_q_values, default=0)
        updated_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * max_future_q_value - current_q_value)
        self.q_table[(self.state_to_tuple(state), action)] = updated_q_value

    def state_to_tuple(self, state):
        """
        Convert a NumPy array state to a hashable tuple for Q-table keys.
        """
        return tuple(map(tuple, state))

    def getQTable(self):
        return self.q_table
