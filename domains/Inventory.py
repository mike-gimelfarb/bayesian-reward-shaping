from domains.Domain import Domain
import numpy as np


L = 7                                       # shelf-life
I = 20                                      # maximum inventory
c = 40.                                     # order cost per unit
K = 20.                                     # cost of promotion
R = [70., 54.3]                             # price per unit
h = 1.                                      # holding cost per unit and time
s = 15.                                     # shortage cost per unit
b = 20.                                     # deterioration cost per unit
P = [[0.2, 0.2, 0.4, 0.2, 0.0, 0.0, 0.0],   # demand distribution [no promote, do promote]
     [0.0, 0.0, 0.0, 0.2, 0.2, 0.4, 0.2]]
N = 100                                     # length of planning horizon


# the domain is borrowed from Chande et al,
# Perishable inventory management and dynamic pricing using RFID technology
# https://pdfs.semanticscholar.org/809a/b16944f5c96e2531450f516df09186c2f9e2.pdf
# with a slight change in parameters
class Inventory(Domain):

    def initial_state(self):
        state = np.zeros(L, dtype='int')
        return state

    def valid_actions(self, state):
        can_order = I - np.sum(state)
        if can_order <= 0:
            return [0, 1]
        total_states = 2 * can_order + 2
        return list(range(int(total_states)))

    def state_dim(self):
        return L

    def action_count(self):
        return (I + 1) * 2

    def act(self, state, action):

        # unpack the action
        promoted = action % 2
        if promoted == 1:
            ordered = (action - 1) / 2
        else:
            ordered = action / 2

        # observe demand
        p = P[promoted]
        k = np.random.choice(len(p), 1, p=p)[0]

        # compute reward
        total = np.sum(state)
        g = R[promoted] * min(k, total) - s * max(k - total, 0) - c * ordered
        reward = g - h * max(total - k, 0) - b * max(state[0] - k, 0) - K * promoted

        # update state for demand
        new_state = np.empty_like(state)
        rem = k
        for i in range(L):
            sold = min(rem, state[i])
            rem -= sold
            new_state[i] = state[i] - sold

        # deteriorate
        new_state[:-1] = new_state[1:]
        new_state[-1] = ordered

        return new_state, reward, False

    def render(self, policy, encoder):
        pass

    def default_encoding(self, state):
        return np.reshape(state, (1, -1))






