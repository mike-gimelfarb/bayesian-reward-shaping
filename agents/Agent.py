'''
Modified on Sep. 22, 2018

@author: Michael Gimelfarb
'''
from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
    
    def __init__(self, state_size, action_size, discount, method):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = discount
        self.method = method

    @abstractmethod
    def backup(self, reward, next_state, next_action, done):
        pass

    @abstractmethod
    def values(self, state):
        pass

    @abstractmethod
    def remember(self, state, action, reward, next_state, next_action, done):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def finish_trial(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    def act_optimal(self, state):
        return np.argmax(self.values(state))
