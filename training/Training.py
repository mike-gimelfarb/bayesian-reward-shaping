'''
Modified on Sep. 22, 2018

@author: Michael Gimelfarb
'''
import random
import math
import numpy as np
import pyprind
import sys
from shaping.RewardShape import RewardShape


class Training:

    def __init__(self, env, agent, encoder):
        self.env = env
        self.agent = agent
        self.encoder = encoder

    def greedy(self, epsilon, state):
        if np.random.rand() <= epsilon:
            return random.randrange(self.agent.action_size)
        else:
            q_values = self.agent.values(state)
            return np.argmax(q_values)

    def run(self, episodes, time_limit, exploration, experts,
            render=False, measure=None, progress=True, online=1, offline=0):

        # reset agent, exploration policy, posterior
        self.agent.reset()
        epsilon = exploration.reset()
        shaping = RewardShape(experts)

        # is our method on or off-policy
        on_policy = self.agent.method == 'sarsa'

        # initialize dict for output data
        data = dict()
        data['epoch'] = []
        data['eps'] = []
        data['measure'] = []
        data['weights'] = [[] for e in shaping.experts]

        # progress bar
        if progress:
            progress_bar = pyprind.ProgBar(episodes, stream=sys.stdout)

        # main outer loop over episodes
        for e in range(episodes):

            # if render
            if render and e % 20 == 0:
                self.env.render(self.agent.act_optimal, self.encoder)

            # re-initialize the environment
            state = self.env.initial_state()
            state_enc = self.encoder(state)

            # fix the weights of the Bayesian expert before the trial
            shaping.fix_weights()
            state_set = []
            reward_set = []

            # when the method is on-policy we need to compute the action
            if on_policy:
                action = self.greedy(epsilon, state_enc)

            # start a new trial
            for _ in range(time_limit):

                # if not on-policy we compute the action here
                if not on_policy:
                    action = self.greedy(epsilon, state_enc)

                # environment performs a transition
                new_state, reward, done = self.env.act(state, action)
                new_state_enc = self.encoder(new_state)

                # select an action according to epsilon-greedy
                if on_policy:
                    new_action = self.greedy(epsilon, new_state_enc)
                else:
                    new_action = None

                # reshape the reward and store memory
                state_set.append(state)
                reward_set.append(reward)
                reward += shaping.bmc(state, new_state, self.agent.gamma)
                self.agent.remember(state_enc, action, reward, new_state_enc, new_action, done)

                # agent replays examples from the memory to train
                for _ in range(online):
                    self.agent.train()

                # update state
                state, state_enc = new_state, new_state_enc
                if on_policy:
                    action = new_action
                if done:
                    break
            
            # agent replays examples from the memory to train
            for _ in range(offline):
                self.agent.train()
                
            # we do this once at the end of the trial
            self.agent.finish_trial()
            epsilon = exploration.update(e)

            # update bayes
            rewards = 0.0
            for i in reversed(range(len(reward_set))):
                rewards = reward_set[i] + self.agent.gamma * rewards
                shaping.update(state_set[i], rewards)
            shaping.fix_weights()

            # prepare the test sample
            measurement, _ = measure(self.env, self.agent.act_optimal, self.encoder)

            # cache the learning progress in a list
            data['epoch'].append(e)
            data['eps'].append(epsilon)
            data['measure'].append(measurement)
            for i, weight in enumerate(shaping.weights):
                data['weights'][i].append(weight)

            # update progressbar
            if progress:
                progress_bar.update()
        return data

    def run_many(self, trials, episodes, time_limit, exploration, experts,
                 measure=None, online=1, offline=0):

        # run trials
        averages = None
        min_len = episodes
        measures = None
        for _ in range(trials):

            # run a trial
            data = self.run(episodes=episodes,
                            time_limit=time_limit,
                            exploration=exploration,
                            experts=experts,
                            render=False,
                            measure=measure,
                            progress=True,
                            online=online,
                            offline=offline)

            # accumulate averages
            if averages is None:
                averages = data
                measures = [[] for i in range(min_len)]
            else:
                for key in averages.keys():
                    if key == 'weights':
                        for i, values in enumerate(averages[key]):
                            averages[key][i] = [x + y for x, y in zip(values, data[key][i])]
                    else:
                        averages[key] = [x + y for x, y in zip(averages[key], data[key])]

            # accumulate series for variance
            for i in range(min_len):
                measures[i].append(data['measure'][i])

        # normalize
        for key in averages.keys():
            if key == 'weights':
                for i, values in enumerate(averages[key]):
                    averages[key][i] = [x / float(trials) for x in values]
            else:
                averages[key] = [x / float(trials) for x in averages[key]]

        # get standard error of reward and measure
        averages['measure_std'] = []
        sqrtn = math.sqrt(trials)
        for i in range(min_len):
            averages['measure_std'].append(np.std(measures[i]) / sqrtn)

        # printing
        for i in range(min_len):
            print('Epoch:\t{}\tMeasure:\t{}\tStd_measure:\t{}\tWeights:\t{}'
                  .format(averages['epoch'][i],
                          averages['measure'][i],
                          round(averages['measure_std'][i], 6),
                          '\t'.join(str(round(w[i], 4)) for w in averages['weights'])))
        return averages
