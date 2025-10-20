# Market entering with variable environment

In this directory, we want to run simulations of market entering for different environment
parameters. We want to check whether the upward bias observed in the baseline experiments
is still present if the environment parameters change.

## Environment parameters

The environment parameters are following:

- sigma: 0.15
- rho: 0.5
- tau: 0.13
- N: 2
- S_c: 1.0 which then give us following equilibrium points:
- nash_payoff: 0.081
- pareto_payoff: 0.364

We do not want to change all these parameters but rather choose which of them will induce
changes in the shape of the reward curve for the entering agent.

We will start with sigma, as it really smoothes out the reward curve.

## Varying sigma

The current value of sigma is 0.15. To vary it, we will use the following values:

- 0.1, 0.3, 0.5, 0.7, 1.

## Plan for each simulation

To run the simulation, we will have to implement a function that computes the nash and
pareto payoffs for the given environment parameters. Then we will run the simulation.
