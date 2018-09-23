'''
Modified on Sep. 22, 2018

@author: Michael Gimelfarb
'''
import numpy as np
import math
from shaping.Average import Average


class RewardShape:

    def __init__(self, experts):

        # initialize all weights to uniform
        # this is the Dirichlet with all parameters equal
        self.experts = experts
        self.count = len(experts)
        self.average = Average()
        self.reset()

    def reset(self):
        self.alpha0 = 1.0 * self.count
        self.params = np.ones(shape=self.count, dtype=float)
        self.evidence = np.zeros(shape=self.count, dtype=float)
        self.mean = (1.0 / self.count) * np.ones(shape=self.count, dtype=float)
        self.average.reset()
        self.weights = None

    def fix_weights(self):

        # fixes the weights used to compute the Bayesian combination
        # shaped rewards so they can be accessed while training
        self.weights = self.mean[:]

    def bmc(self, state, next_state, df):

        # computes the expert shapes for the given state -> state1
        # transitions, weighted by the posterior weights
        weights = self.weights
        experts = self.experts
        shape, next_shape = 0.0, 0.0
        for i in range(self.count):
            next_shape += weights[i] * experts[i](next_state)
            shape += weights[i] * experts[i](state)
        return df * next_shape - shape

    def update(self, state, reward):

        # update the tuning variance parameter for the gaussian
        var = self.average.update(reward)
        if var == 0:
            return

        # given the current evidence, perform approximate bayes rule using Dirichlet projection
        for i in range(self.count):
            expert_value = self.experts[i](state)
            delta = reward - expert_value
            log = -0.5 * delta * delta / var - 0.5 * math.log(2.0 * math.pi * var)
            self.evidence[i] = math.exp(log)
        self.update_from_evidence()

    def update_from_evidence(self):
        if self.count == 1:
            return

        # this is the main subroutine to compute the Dirichlet projection
        # first we compute the new Dirichlet means and variance by moment matching
        params, mean, evidence = self.params, self.mean, self.evidence
        a0 = self.alpha0
        ca = np.dot(evidence, params)
        if ca == 0.0:
            return

        # choose the index for the second moment with largest denominator
        denom_max = 0.0
        m1, s1 = None, None
        for i in range(self.count):
            m11 = params[i] * (evidence[i] + ca) / (ca * (a0 + 1.0))
            s11 = params[i] * (params[i] + 1.0) * (ca + 2.0 * evidence[i]) / \
            (ca * (a0 + 1.0) * (a0 + 2.0))
            denom = s11 - m11 * m11
            if denom > denom_max or i == 0:
                m1, s1 = m11, s11
                denom_max = denom
        if m1 == s1 or s1 == m1 * m1:
            return

        # next, we update the alpha parameters of the Dirichlet
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
