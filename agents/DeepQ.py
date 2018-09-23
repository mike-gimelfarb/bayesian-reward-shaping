'''
Modified on Sep. 22, 2018

@author: Michael Gimelfarb
'''
import random
import numpy as np
from collections import deque
from agents.Agent import Agent


class DeepQ(Agent):

    def __init__(self, state_size, action_size, discount,
                 build_model, batch_size, epochs, memory_size):
        super().__init__(state_size, action_size, discount, 'q')
        self.state_size = state_size
        self.action_size = action_size
        self.build_model = build_model
        self.batch_size = batch_size
        self.epochs = epochs
        self.memory_size = memory_size
        
        # get the architecture
        self.memory = deque(maxlen=memory_size)
        self.reset()

    def backup(self, reward, new_state, _, done):
        if done:
            return reward 
        else:
            return reward + self.gamma * np.amax(self.values(new_state))

    def values(self, state):
        return self.model.predict(state)[0]

    def remember(self, state, action, reward, next_state, next_action, done):
        self.memory.append((state, action, reward, next_state, next_action, done))

    def train(self):

        # we don't have enough memory for training
        batch_size = self.batch_size
        if len(self.memory) < self.batch_size:
            return

        # sample a minibatch
        state_size = self.state_size
        mini_batch = random.sample(self.memory, batch_size)
        states = np.zeros((batch_size, state_size), dtype=float)
        next_states = np.zeros((batch_size, state_size), dtype=float)
        for i, memory in enumerate(mini_batch):
            state, _, _, new_state, _, _ = memory
            states[i] = state
            next_states[i] = new_state

        # make the predictions with current models
        values = self.model.predict(states)
        next_values = self.model.predict(next_states)

        # update model weights based on error in prediction
        gamma = self.gamma
        for i, memory in enumerate(mini_batch):
            _, action, reward, _, _, done = memory
            values[i][action] = reward
            if not done:
                values[i][action] += gamma * np.amax(next_values[i])

        # make batch which includes target q value and predicted q value
        self.model.fit(states, values,
                       batch_size=batch_size,
                       epochs=self.epochs,
                       verbose=0)

    def reset(self):
        self.memory = deque(maxlen=self.memory.maxlen)
        self.model = self.build_model()

    def finish_trial(self):
        pass
