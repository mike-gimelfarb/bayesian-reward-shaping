from abc import ABC, abstractmethod
import numpy as np


# An abstract agent integrating learning, memory and prediction/control
# for stochastic control problems.
class Agent(ABC):

    def __init__(self, state_size, action_size, discount):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = discount
        self.memory = None

    @abstractmethod
    def action_values(self, state):
        return np.zeros(self.action_size, dtype='float')

    @abstractmethod
    def remember(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def replay(self, method, eps):
        pass

    @abstractmethod
    def reset(self):
        pass

    def act_optimal(self, state):
        values = self.action_values(state)
        return np.argmax(values)

    def backup(self, reward, next_state, done, method, eps):

        # the Bellman back-ups are either SARSA or Q-learning
        # see the book by Sutton and Barto for more details
        if done:
            return reward
        values = self.action_values(next_state)
        if method == 'sarsa':
            next_value = eps * np.average(values) + (1.0 - eps) * np.amax(values)
        else:
            next_value = np.amax(values)
        return reward + self.gamma * next_value
