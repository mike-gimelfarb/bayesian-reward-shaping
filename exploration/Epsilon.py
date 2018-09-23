'''
Modified on Sep. 22, 2018

@author: Michael Gimelfarb
'''
from abc import ABC, abstractmethod


class Epsilon(ABC):

    def __init__(self, initial_epsilon):
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon

    @abstractmethod
    def next_epsilon(self, current_epsilon, epoch):
        pass

    def update(self, epoch):
        self.epsilon = self.next_epsilon(self.epsilon, epoch)
        return self.epsilon

    def reset(self):
        self.epsilon = self.initial_epsilon
        return self.epsilon
