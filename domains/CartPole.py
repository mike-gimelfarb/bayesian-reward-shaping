import gym
import numpy as np
from domains.Domain import Domain


# note that the Cart Pole domain is based on the open ai gym project
# this class simply acts as a wrapper for this domain
class CartPole(Domain):

    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.env.reset()

    def act(self, state, action):
        new_state, reward, done, _ = self.env.step(action)
        reward = 1.0 if not done else 0.0
        return new_state, reward, done

    def valid_actions(self, state):
        return [0, 1]

    def initial_state(self):
        return self.env.reset()

    def state_dim(self):
        return self.env.observation_space.shape[0]

    def action_count(self):
        return self.env.action_space.n

    def render(self, policy, encoder):
        act = self.act
        state = self.initial_state()
        done = False
        while not done:
            self.env.render()
            state_enc = encoder(state)
            action = policy(state_enc)
            new_state, reward, done = act(state, action)
            state = new_state

    def default_encoding(self, state):
        return np.reshape(state, [1, self.state_dim()])

