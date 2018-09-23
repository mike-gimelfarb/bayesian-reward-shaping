'''
Modified on Sep. 22, 2018

@author: Michael Gimelfarb
'''
from abc import ABC, abstractmethod


class Domain(ABC):

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def initial_state(self):
        pass

    @abstractmethod
    def valid_actions(self):
        pass

    @abstractmethod
    def act(self, state, action):
        pass

    @abstractmethod
    def render(self, policy, encoder):
        pass

    @abstractmethod
    def default_encoding(self, state):
        pass

    def rewards(self, steps, policy, df, encoder):
        act = self.act
        state = self.initial_state()
        factor = 1.0
        rewards = 0.0
        steps_to_goal = 0
        for _ in range(steps):
            state_enc = encoder(state)
            action = policy(state_enc)
            new_state, reward, done = act(state, action)
            rewards += factor * reward
            steps_to_goal += 1
            factor *= df
            state = new_state
            if done:
                break
        return state, rewards, steps_to_goal

    def print_trace(self, steps, policy, encoder):
        act = self.act
        state = self.initial_state()
        print('step\tstate\taction')
        for time in range(steps):
            state_enc = encoder(state)
            action = policy(state_enc)
            print(str(time) + '\t' + str(state) + '\t' + str(action))
            state, _, done = act(state, action)
            if done:
                break
