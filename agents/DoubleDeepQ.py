'''
Modified on Sep. 22, 2018

@author: Michael Gimelfarb
'''
import random
import numpy as np
from collections import deque
from agents.Agent import Agent


class DoubleDeepQ(Agent):

    def __init__(self, state_size, action_size, discount,
                 build_model, batch_size, epochs, memory_size):
        super().__init__(state_size, action_size, discount, 'q')

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.discount = discount

        # get the architectures for targets and values
        self.build_model = build_model
        self.model = build_model()
        self.target_model = build_model()

        # replay memory and learning
        self.batch_size = batch_size
        self.epochs = epochs
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)

    def backup(self, reward, new_state, new_action, done):
        if done:
            return reward
        else:
            return reward + self.gamma * np.amax(self.values(new_state))

    def values(self, state):
        return self.model.predict(state)[0]

    def remember(self, state, action, reward, next_state, next_action, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):

        # we don't have enough memory for training
        batch_size = self.batch_size
        if len(self.memory) < batch_size:
            return

        # sample a mini-batch
        state_size = self.state_size
        mini_batch = random.sample(self.memory, batch_size)
        update_input = np.zeros((batch_size, state_size), dtype=float)
        update_target = np.zeros((batch_size, state_size), dtype=float)
        actions, rewards, dones = [], [], []
        for i, memory in enumerate(mini_batch):
            state, action, reward, next_state, done = memory
            update_input[i] = state
            actions.append(action)
            rewards.append(reward)
            update_target[i] = next_state
            dones.append(done)

        # make the predictions with current models
        target = self.model.predict(update_input)
        target_next = self.model.predict(update_target)
        target_val = self.target_model.predict(update_target)

        # update model weights based on error in prediction
        gamma = self.discount
        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                a = np.argmax(target_next[i])
                target[i][actions[i]] = rewards[i] + gamma * target_val[i][a]

        # make mini-batch which includes target q value and predicted q value
        self.model.fit(update_input, target,
                       batch_size=batch_size,
                       epochs=self.epochs,
                       verbose=0)

    def reset(self):
        self.memory = deque(maxlen=self.memory.maxlen)
        self.model = self.build_model()
        self.target_model = self.build_model()

    def finish_trial(self):
        self.target_model.set_weights(self.model.get_weights())
