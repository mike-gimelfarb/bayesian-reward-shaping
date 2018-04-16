import random
from shaping.RewardShape import RewardShape


# This training loop is based on the algorithm (Algorithm 1) provided in
# Mnih et al, Playing Atari with Deep Reinforcement Learning
# https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
# some significant generalizations have been made:
# 1. this supports both online and offline batch learning
# 2. the method can be Q-learning or SARSA
# 3. supports custom exploration policies with general epsilon decay strategies
# 4. accepts any Keras network (note: currently this should work for Dense and CNN networks)
# 5. supports averaging over multiple trials for academic research
class Trainer:

    def __init__(self, env, agent, encoder):
        self.env = env
        self.agent = agent
        self.encoder = encoder

    def run(self, epochs, horizon, exploration, experts,
            render_every=0, print_every=1,
            method='q', replay_online_every=1, replays_offline=0,
            measure=None):

        # initialization
        game = self.env
        agent = self.agent
        phi = self.encoder
        df = agent.gamma
        agent.reset()
        epsilon = exploration.reset()
        shaping = RewardShape(experts)

        # initialize dict for output data
        data = dict()
        data['epoch'] = []
        data['episodes'] = []
        data['rewards'] = []
        data['eps'] = []
        data['measure'] = []
        data['weights'] = [[] for e in shaping.experts]

        ####################################################################################
        #                          BEGINNING OF TRAINING LOOP                              #
        ####################################################################################
        # run each episode sequentially
        for epoch in range(epochs):

            # if render
            if epoch > 0 and render_every > 0 and epoch % render_every == 0:
                game.render(agent.act_optimal, phi)

            # re-initialize the environment
            state = game.initial_state()
            state_enc = phi(state)

            # fix the weights of the Bayesian expert before the epoch
            # we use TD(0) return as the reward estimate in Bayes' rule
            # so that there is no direct "dependence" on the value function
            # which is a noisy estimate of the returns
            td0 = 0.0
            shaping.fix_weights()

            # for the reward estimation
            steps = 0
            rewards = 0.0

        ####################################################################################
        #                               BEGINNING OF EPOCH                                 #
        ####################################################################################
            # run each step of the epoch sequentially
            for episode in range(horizon):

                # select an action according to epsilon-greedy
                actions = game.valid_actions(state)
                if random.random() < epsilon:
                    action = random.choice(actions)
                else:
                    action = agent.act_optimal(state_enc)

                # environment performs a transition
                new_state, reward, done = game.act(state, action)
                new_state_enc = phi(new_state)

                # update the Bayesian agent
                td0 = reward + df * td0
                shaping.update(state, td0)

                # update reward estimate
                steps += 1
                rewards = reward + df * rewards

                # re-shape the reward
                reward += shaping.bmc(state, new_state, df)

                # store in replay memory the state action pair
                agent.remember(state_enc, action, reward, new_state_enc, done)

                # agent replays examples from the memory to train
                if replay_online_every > 0 and episode % replay_online_every == 0:
                    agent.replay(method, epsilon)

                # update state
                state = new_state
                state_enc = new_state_enc
                if done:
                    break

        ####################################################################################
        #                          END OF EPOCH - MODEL UPDATE                             #
        ####################################################################################
            # agent replays examples from the memory to train
            if replays_offline > 0:
                for i in range(replays_offline):
                    agent.replay(method, epsilon)

            # anneal the exploration rate
            epsilon = exploration.update(epoch)

            # prepare the custom measurement and stopping criterion
            if measure is None:
                measurement, stop = 0.0, False
            else:
                measurement, stop = measure(game, agent.act_optimal, phi)

        ####################################################################################
        #                   PRINT PROGRESS HERE AND FINALIZE EPOCH                         #
        ####################################################################################
            if print_every > 0 and epoch % print_every == 0:

                # check the quality of the policy and print progress
                str_weights = '\t'.join(str(round(w, 4)) for w in shaping.weights)
                print("Epoch:\t{}\tEpisodes:\t{}\tRewards:\t{}"
                      "\tExploration:\t{}\tMeasure:\t{}\tWeights:\t{}"
                      .format(epoch, steps, round(rewards, 6), round(epsilon, 6),
                              measurement, str_weights))

            # cache the learning progress in a list
            data['epoch'].append(epoch)
            data['episodes'].append(steps)
            data['rewards'].append(rewards)
            data['eps'].append(epsilon)
            data['measure'].append(measurement)
            for i, weight in enumerate(shaping.weights):
                data['weights'][i].append(weight)

            # check if we need to stop early
            if stop:
                if print_every > 0:
                    print("Stopped at epoch {} with reward {} and measurement {}."
                          .format(epoch, round(rewards, 6), measurement))
                return data

        # did not achieve the desired measurement
        if print_every > 0:
            print("Did not achieve the desired measure after {} epochs. "
                  "Please run the code again or tune some parameters."
                  .format(epochs))
        return data

    def run_many(self, trials, epochs, horizon, exploration, experts,
                 method='q', replay_online_every=1, replays_offline=0,
                 measure=None):

        # run trials
        averages = None
        for trial in range(trials):

            # run a trial
            data = self.run(epochs=epochs, horizon=horizon,
                            exploration=exploration, experts=experts,
                            render_every=0, print_every=0,
                            method=method,
                            replay_online_every=replay_online_every,
                            replays_offline=replays_offline,
                            measure=measure)

            # accumulate averages
            if averages is None:
                averages = data
            else:
                for key in averages.keys():
                    if key == 'weights':
                        for i, values in enumerate(averages[key]):
                            averages[key][i] = [x + y for x, y in zip(values, data[key][i])]
                    else:
                        averages[key] = [x + y for x, y in zip(averages[key], data[key])]

            # print progress
            print('Trial ' + str(trial) + ' complete.')

        # normalize
        for key in averages.keys():
            if key == 'weights':
                for i, values in enumerate(averages[key]):
                    averages[key][i] = [x / float(trials) for x in values]
            else:
                averages[key] = [x / float(trials) for x in averages[key]]
        return averages
