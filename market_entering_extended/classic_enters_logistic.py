"""This file contains games of non-stationary agents between each other where one
agent is fully converged.


These simulations are extended versions of the simulations and include variable 
logistic sliding window tau.

In this file, the script runs pricing competitions where the classic agent enters
 the game with the logistic agent already present and fully converged.

In between runs, the tau for the logistic agent is to be varied which generates
different results and allows analysis of the influence of the sliding window
of the logistic agent on the evolution of pricing.
"""

# Import packages
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import os


from bandits import EpsGreedy
from models import NonStationaryClassicModel, NonStationaryLogisticModel

# Import the utility functions for market entering
from market_entering_extended import me_utils

logger = ut.get_logger(__name__)

# Read common parameters from yaml file
params = ut.read_parameters('common_parameters.yaml')
logger.info(json.dumps(params, indent=4))
nash_payoff = params['environment_parameters']['nash_payoff']
pareto_payoff = params['environment_parameters']['pareto_payoff']


def dimless_payoff(x):
    return (x - nash_payoff) / (pareto_payoff - nash_payoff)


# Set seed
np.random.seed(0)


# Define parameters
no_sim = 100  # Number of simulations
T = 2000  # Number of time steps
no_actions = 129  # Number of actions
action_set = np.linspace(0, 0.9, no_actions, endpoint=True)


# Initialize the agents
tau_logistic = 40  # Window size for sliding window method - THIS IS VARIABLE BETWEEN RUNS
tau_classic = 40
epsilon = 0.05  # Epsilon for epsilon-greedy bandit
variance = 1.
dimension = 2

# Plotting parameters
results_folder = f'results/market_entering/classic_enters_logistic_no_actions_{no_actions}_T_{T}_tau_{tau_logistic}'


# Initialize the model
model_1 = NonStationaryLogisticModel(
    dimension=dimension, candidate_margins=action_set, method='sliding_window', tau=tau_logistic)

# Initialize epsilon-greedy bandit
bandit_1 = EpsGreedy(eps=epsilon, model=model_1)


# Initialize the model
model_2 = NonStationaryClassicModel(
    variance=variance, candidate_margins=action_set, method='sliding_window', tau=tau_classic)
# Initialize epsilon-greedy bandit
bandit_2 = EpsGreedy(eps=epsilon, model=model_2)

# Define simulation setup
simulation_setup = {
    'action_set': action_set,
    'no_sim': no_sim,
    'T': T
}

# Run simulations
me_utils.simulate_game(bandit_1, bandit_2, results_folder,
                       'Logistic Eps-Greedy', 'Classic Eps-Greedy', simulation_setup)
