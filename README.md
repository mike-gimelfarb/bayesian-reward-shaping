# Reinforcement Learning with Multiple Experts

Bayesian Reward Shaping Ensemble Framework for Deep Reinforcement Learning. 

## Description

This small and fairly self-contained (see prerequisites below) package accompanies an article published in Advances in Neural Information Processing Systems (NeurIPS) entitled "Reinforcement Learning with Multiple Experts: A Bayesian Model Combination Approach" in December of 2018. 

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
4. Removed jupyter notebooks containing outdated experiments.

## Citation

To cite the framework:

> @inproceedings{gimelfarb2018reinforcement,  
>  title={Reinforcement Learning with Multiple Experts: A Bayesian Model Combination Approach},  
>  author={Gimelfarb, Michael and Sanner, Scott and Lee, Chi-Guhn},  
>  booktitle={Advances in Neural Information Processing Systems},  
>  pages={9549--9559},  
>  year={2018}
> }

## References

[1] Gimelfarb, Michael, Scott Sanner, and Chi-Guhn Lee. "Reinforcement Learning with Multiple Experts: A Bayesian Model Combination Approach." Advances in Neural Information Processing Systems. 2018.
[2] Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep Reinforcement Learning with Double Q-Learning." AAAI. Vol. 2. 2016.
[3] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).
[4] Ng, Andrew Y., Daishi Harada, and Stuart Russell. "Policy invariance under reward transformations: Theory and application to reward shaping." ICML. Vol. 99. 1999.
[5] Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 1998.
