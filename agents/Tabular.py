'''
Modified on Sep. 22, 2018

@author: Michael Gimelfarb
'''
from agents.Agent import Agent
import numpy as np


class Tabular(Agent):

    def __init__(self, state_size, action_size, discount, method, learning_rate, 
                 lr_decay=1.0):
        super().__init__(state_size, action_size, discount, method)
        self.alpha = learning_rate
        self.alpha0 = learning_rate
        self.lr_decay = lr_decay
        self.Q = None
        self.experience = None
        self.reset()

    def backup(self, reward, new_state, new_action, done):
        if done:
            return reward
        elif self.method == 'q':
            return reward + self.gamma * np.amax(self.values(new_state))
        else:
            return reward + self.gamma * self.values(new_state)[new_action]

    def values(self, state):
        if state in self.Q:
            return self.Q[state]
        else:
            return np.zeros(self.action_size, dtype=float)

    def remember(self, state, action, reward, next_state, next_action, done):
        self.experience = (state, action, reward, next_state, next_action, done)

    def train(self):
        state, action, reward, next_state, next_action, done = self.experience

        # set table entry to zeros if this state hasn't been visited
        if state not in self.Q:
            self.Q[state] = np.zeros(self.action_size, dtype=float)

        # compute target and prediction and update
        prediction = self.Q[state][action]
        target = self.backup(reward, next_state, next_action, done)
        error = target - prediction
        self.Q[state][action] += self.alpha * error
        return self.Q[state][action]

    def finish_trial(self):
        self.alpha *= self.lr_decay

    def reset(self):
        self.Q = {}
        self.experience = None
        self.alpha = self.alpha0
