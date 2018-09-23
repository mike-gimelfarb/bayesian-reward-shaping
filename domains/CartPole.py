'''
Modified on Sep. 22, 2018

@author: Michael Gimelfarb
'''
import gym
import math
import numpy as np
from domains.Domain import Domain


class CartPole(Domain):

    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.env.reset()
        self.buckets = (3, 3, 6, 3)

    def copy(self):
        return CartPole()

    def act(self, state, action):
        new_state, _, done, _ = self.env.step(action)
        reward = 1.0 if not done else 0.0
        return new_state, reward, done

    def valid_actions(self):
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
            new_state, _, done = act(state, action)
            state = new_state

    def default_encoding(self, state):
        res = np.reshape(state, (1, -1))
        return res

    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5,
                        self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5,
                        self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / 
                  (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)
