import numpy as np
import math
from shaping.Average import Average


class RewardShape:

    def __init__(self, experts):

        # initialize all weights to uniform
        # this is the dirichlet with all parameters equal
        self.experts = experts
        self.count = len(experts)
        self.alpha0 = 1.0 * self.count
        self.params = np.ones(shape=self.count)
        self.evidence = np.zeros(shape=self.count)
        self.mean = (1.0 / self.count) * np.ones(shape=self.count)
        self.average = Average()
        self.weights = None

    def reset(self):
        self.alpha0 = 1.0 * self.count
        self.params = np.ones(shape=self.count)
        self.evidence = np.zeros(shape=self.count)
        self.mean = (1.0 / self.count) * np.ones(shape=self.count)
        self.average.reset()
        self.weights = None

    def fix_weights(self):

        # fixes the weights used to compute the bayesian combination
        # shaped rewards so they can be accessed while training
        self.weights = self.mean[:]

    def bmc(self, state, next_state, df):

        # computes the expert shapes for the given state -> state1
        # transitions, weighted by the posterior weights
        weights, experts = self.weights, self.experts
        shape = next_shape = 0.0
        for i in range(self.count):
            next_shape += weights[i] * experts[i](next_state)
            shape += weights[i] * experts[i](state)
        return df * next_shape - shape

    def update(self, state, reward):

        # update the tuning variance parameter for the gaussian
        var = self.average.update(reward)

        # given the current evidence, perform approximate bayes rule
        # with dirichlet projection
        experts = self.experts
        evidence = self.evidence
        for i in range(self.count):
            delta = reward - experts[i](state)
            if var > 0.0:
                log = -delta * delta / (2.0 * var) - 0.5 * math.log(2.0 * math.pi * var)
                evidence[i] = math.exp(log)
        self.update_from_evidence()

    def update_from_evidence(self):

        # this is the main subroutine to compute the dirichlet projection
        # first we compute the new dirichlet means and variance by moment matching
        params, mean, evidence = self.params, self.mean, self.evidence
        a0 = self.alpha0
        ca = np.dot(evidence, params)
        if ca == 0.0:
            return
        m1 = params[0] * (evidence[0] + ca) / (ca * (a0 + 1.0))
        s1 = params[0] * (params[0] + 1.0) * (ca + 2.0 * evidence[0]) / (ca * (a0 + 1.0) * (a0 + 2.0))
        if m1 == s1 or s1 == m1 * m1:
            return

        # next, we update the alpha parameters of the dirichlet
        # the relevant formulae to do this are provided in the paper
        alpha01 = (m1 - s1) / (s1 - m1 * m1)
        s = 0.0
        for k in range(self.count - 1):
            mk = params[k] * (evidence[k] + ca) / (ca * (a0 + 1.0))
            params[k] = alpha01 * mk
            s += params[k]
            mean[k] = params[k] / alpha01
        params[-1] = alpha01 - s
        mean[-1] = params[-1] / alpha01
        self.alpha0 = alpha01
