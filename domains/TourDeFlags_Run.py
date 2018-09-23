'''
Modified on Sep. 22, 2018

@author: Michael Gimelfarb
'''
from domains.TourDeFlags import TourDeFlags
from exploration.DecayingEpsilon import DecayingEpsilon
from agents.Tabular import Tabular
from agents.DeepQ import DeepQ
from training.Training import Training

import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.optimizers import Adam


def main():

    # set up the domain
    maze = np.array([[0, 0, 0, 0, 5],
                     [0, 2, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [3, 0, 0, 0, 1],
                     [0, 0, 0, 0, 4]])
    domain = TourDeFlags(maze, (4, 0))

    # See "Policy invariance under reward transformations:
    # Theory and application to reward shaping" (Ng et al., 1999)
    zero_shape = lambda state: 0.0
    heuristic_shape = lambda state:-22.0 * ((5.0 - state[2] - 0.5) / 5.0)
    bad_shape = lambda state:-heuristic_shape(state)
    random_shape = lambda state: random.random() * 40.0 - 20.0

    def good_shape(state):
        row, col, flags = state
        if flags == 0:
            return -abs(row - 3.0) - abs(col - 4.0) - 17.0
        elif flags == 1:
            return -abs(row - 1.0) - abs(col - 1.0) - 12.0
        elif flags == 2:
            return -abs(row - 3.0) - abs(col - 0.0) - 9.0
        elif flags == 3:
            return -abs(row - 4.0) - abs(col - 4.0) - 4.0
        elif flags == 4:
            return -abs(row - 0.0) - abs(col - 4.0)
        else:
            return 0.0

    # set up the experts
    experts = [zero_shape, heuristic_shape, good_shape, random_shape, bad_shape]

    # we will define the success criterion and stopping rule
    def steps_to_goal(tdf, policy, enc):
        _, _, steps = tdf.rewards(tdf.max_steps, policy, 1.0, enc)
        return steps, False

    # decide here whether we will use deep learning or tabular
    use_deep = True
    
    if use_deep:

        # one-hot encoding
        encoding = domain.default_encoding

        # set up the neural network as the function approximator
        input_dim = domain.width + domain.height + (1 + domain.goals)
        
        def network_initializer():
            model = Sequential()
            model.add(Dense(25, input_shape=(input_dim,)))
            model.add(LeakyReLU())
            model.add(Dense(25))
            model.add(LeakyReLU())
            model.add(Dense(4,
                            activation='linear'))
            model.compile(optimizer=Adam(0.001),
                          loss='mse')
            return model

        # set up the learning agent
        agent = DeepQ(state_size=input_dim, action_size=4,
                      discount=1.0,
                      build_model=network_initializer,
                      batch_size=16, epochs=5,
                      memory_size=10000)

        # set up the exploration schedule for the greedy epsilon policy
        eps_schedule = DecayingEpsilon(0.08, 0.08, 1.0)
        
    else:

        # no encoding
        encoding = lambda state: state

        # set up the tabular
        agent = Tabular(state_size=3, action_size=4,
                        method='sarsa',
                        discount=1.0,
                        learning_rate=0.36, lr_decay=1.0)

        # set up the exploration schedule for the greedy epsilon policy
        eps_schedule = DecayingEpsilon(1.0, 0.0, 0.98)

    # finally create the training algorithm
    training = Training(domain, agent, encoding)

    # we start by training for 1 trial on all the experts and sarsa
    _ = training.run_many(trials=1, episodes=200, time_limit=200,
                          exploration=eps_schedule,
                          experts=experts,
                          measure=steps_to_goal,
                          online=1, offline=0)


if __name__ == "__main__":
    main()
