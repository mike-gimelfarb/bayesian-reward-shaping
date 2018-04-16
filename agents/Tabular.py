from collections import deque
from agents.Agent import Agent
import numpy as np


class Tabular(Agent):

    def __init__(self, state_size, action_size, discount):
        super().__init__(state_size, action_size, discount)
        self.Q = None
        self.memory = None
        self.learning_rate = None

    def setup_table(self, lr, memory):
        self.learning_rate = lr
        self.memory = deque(maxlen=memory)
        self.Q = {}

    def action_values(self, state):
        state = tuple(state)
        if state in self.Q:
            return self.Q[state]
        else:
            return np.zeros(self.action_size, dtype='float')

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, method, eps):

        # the replay just plays all examples in memory
        # to be consistent to how learning is done for the deep network
        q = self.Q
        alpha = self.learning_rate
        for (state, action, reward, next_state, done) in self.memory:
            state = tuple(state)
            if state not in q:
                q[state] = np.zeros(self.action_size)
            target = self.backup(reward, next_state, done, method, eps)
            q[state][action] += alpha * (target - q[state][action])

    def reset(self):
        self.Q = {}
        self.memory = deque(maxlen=self.memory.maxlen)

