import random
import numpy as np
from keras import backend as backend
from collections import deque
from agents.Agent import Agent


class DeepAgent(Agent):

    def __init__(self, state_size, action_size, discount):
        super().__init__(state_size, action_size, discount)
        self.batch_size = None
        self.epochs_per_batch = None
        self.model = None

    def setup_nn(self, batch_size, epochs_per_batch, memory_size, network):
        self.batch_size = batch_size
        self.epochs_per_batch = epochs_per_batch
        self.memory = deque(maxlen=memory_size)
        self.model = network

    def action_values(self, state):
        return self.model.predict(state)[0]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, method, eps, batch_size=16):
        if len(self.memory) < self.batch_size:
            return
        nn = self.model
        batch = random.sample(self.memory, self.batch_size)
        inputs = np.zeros((self.batch_size, self.state_size))
        targets = np.zeros((self.batch_size, self.action_size))
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            inputs[i] = state
            targets[i] = nn.predict(state)
            targets[i, action] = self.backup(reward, next_state, done, method, eps)
        nn.fit(inputs, targets,
               epochs=self.epochs_per_batch,
               batch_size=batch_size,
               verbose=0)

    def reset(self):
        self.memory = deque(maxlen=self.memory.maxlen)

        # this code comes from
        # https://www.codementor.io/nitinsurya/how-to-re-initialize-keras-model-weights-et41zre2g
        # for resetting the neural network weights
        session = backend.get_session()
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)
