# Reinforcement Learning with Multiple Experts

Bayesian Reward Shaping Ensemble Framework for Deep Reinforcement Learning. 

## Description

This small and fairly self-contained (see prerequisites below) package accompanies an upcoming article to be published in Advances in Neural Information Processing Systems (NIPS) entitled "Reinforcement Learning with Multiple Experts: A Bayesian Model Combination Approach" (the article link will be provided soon).

This packages provides an online and efficient Bayesian ensemble algorithm for potential-based reward shaping by combining multiple experts (potential function). 

## Prerequisites

Tested on Python 3.5 with standard packages (e.g. numpy) and the following additional packages:

1. Keras with tensorflow backend
2. OpenAI Gym for the Cartpole implementation
3. pyprind for tracking progress of convergence

## Recent Changes

The new version includes the following bug fixes:

1. Fixed a critical error in the training loop (the Monte Carlo estimate for updating the posterior over experts was computed in the incorrect order)
2. Added Double DQN agent
3. Added arguments for learning rate decay for tabular methods
4. Removed jupyter notebooks containing outdated experiments - new notebooks with updated experiments as presented in the paper should be added in the near future

## References

- Gimelfarb, Michael, et al. “Reinforcement Learning with Multiple Experts: A Bayesian Model Combination Approach.” Advances in Neural Information Processing Systems (NIPS). Forthcoming.
- Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep Reinforcement Learning with Double Q-Learning." AAAI. Vol. 2. 2016.
- Kunuth, D. "The Art of Computer Programming vol. 2 Seminumerical Algorithms." (1998).
- Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).
- Ng, Andrew Y., Daishi Harada, and Stuart Russell. "Policy invariance under reward transformations: Theory and application to reward shaping." ICML. Vol. 99. 1999.
- Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 1998.
