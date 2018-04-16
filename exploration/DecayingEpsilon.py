from exploration.Epsilon import Epsilon


class DecayingEpsilon(Epsilon):

    def __init__(self, initial_epsilon, min_epsilon, epsilon_decay):
        super().__init__(initial_epsilon)
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

    def next_epsilon(self, current_epsilon, epoch):
        if current_epsilon > self.min_epsilon:
            return current_epsilon * self.epsilon_decay
        else:
            return current_epsilon

