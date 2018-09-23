'''
Modified on Sep. 22, 2018

@author: Michael Gimelfarb
'''
from domains.CartPole import CartPole
from exploration.DecayingEpsilon import DecayingEpsilon
from agents.DeepQ import DeepQ
from agents.Tabular import Tabular
from training.Training import Training

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.regularizers import l2

FIFTEEN_DEGREES = 0.26179938779

discount = 0.95
df = 1.0 / (1.0 - discount)


def main():
    
    # create the domain
    domain = CartPole()

    # use default encoding for the states
    encoding = domain.default_encoding

    # let's import the pre-trained networks to use as the experts
    model_good = keras.models.load_model(
        '/home/michael/eclipse-workspace/RewardShaping/domains/cartpole_model_6_by_2.h5')
    
    def predict(state):
        state = np.reshape(state, (1, -1))
        return model_good.predict(state)[0]

    # now we will define the shaping functions from these networks
    ok_shape = lambda state: df * (1.0 - abs(state[2]) / FIFTEEN_DEGREES)
    good_shape = lambda state: np.amax(predict(state))
    bad_shape = lambda state:-good_shape(state)
    random_shape = lambda state: np.random.random() * 2.0 * df - df
    zero_shape = lambda state: 0.0

    # we start by training for 10 trials on all the experts and sarsa
    experts = [zero_shape, ok_shape, good_shape, bad_shape, random_shape]

    # we will define the success criterion and stopping rule
    def steps_balanced(cartpole, policy, enc):
        _, _, steps = cartpole.rewards(500, policy, discount, enc)
        return steps, False

    # decide here whether we will use deep learning or tabular
    use_deep = True
    
    if use_deep:

        # set up the neural network as the function approximator
        def build_model():
            model = Sequential()
            model.add(Dense(12, input_dim=4,
                            activation='relu',
                            kernel_regularizer=l2(1e-6)))
            model.add(Dense(12,
                            activation='relu',
                            kernel_regularizer=l2(1e-6)))
            model.add(Dense(2,
                            activation='linear',
                            kernel_regularizer=l2(1e-6)))
            model.compile(loss='mse',
                          optimizer=Adam(lr=0.0005))
            return model

        # set up the exploration schedule for the greedy epsilon policy
        eps_schedule = DecayingEpsilon(1.0, 0.01, 0.98)

        # set up the learning agent
        agent = DeepQ(state_size=4, action_size=2,
                      discount=discount,
                      build_model=build_model,
                      batch_size=32, epochs=1,
                      memory_size=10000)
        
        # finally create the training algorithm
        training = Training(domain, agent, encoding)
    
        # train
        _ = training.run_many(trials=1, episodes=300, time_limit=500,
                              exploration=eps_schedule,
                              experts=experts,
                              measure=steps_balanced,
                              online=0, offline=100)
        
    else:
        
        encoding = domain.discretize
        agent = Tabular(state_size=4, action_size=2,
                        discount=discount,
                        method='q',
                        learning_rate=0.5, lr_decay=0.99)

        # set up the exploration schedule for the greedy epsilon policy
        eps_schedule = DecayingEpsilon(1.0, 0.01, 0.98)
        
        # finally create the training algorithm
        training = Training(domain, agent, encoding)
    
        # train
        _ = training.run_many(trials=1, episodes=300, time_limit=500,
                              exploration=eps_schedule,
                              experts=experts,
                              measure=steps_balanced,
                              online=1, offline=0)


if __name__ == "__main__":
    main()
