"""This file contains games of non-stationary agents between each other where one
agent is fully converged.

In this script, the logistic agent enters the game with the classic agent.
The agents are non-stationary and the classic agent is fully converged.

In this file specifically, we test addition of prior information to the logistic agent.

The prior information is implemented by feeding the logistic agent with fake observations
that encode that:
1. Case 1: The probability of conversion at -1 is 1 and at 2 is 0. (suffix = 'both')
2. Case 2: The probability of conversion at 2 is 0. (suffix = 'right')

The cases are differentation by commenting the apprioriate lines in the code.
"""

# Import packages
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import os

from copy import deepcopy

from bandits import EpsGreedy
from models import NonStationaryClassicModel, NonStationaryLogisticModel

# Initialize the environment
from envs import InsuranceMarketCt

from prior_me import me_utils

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

# Define the fake observations
no_obs = 5  # Number of pairs of observations to be fed to the model
# Indices of the actions for the margin map
action_indices = np.array([0, 1]*no_obs)
margin_map = {0: -1, 1: 2.}
observation_map = {0: 1., 1: 0.}
fake_margins = np.array(
    list(map(lambda x: margin_map[x], action_indices)))
fake_observations = np.array(
    list(map(lambda x: observation_map[x], action_indices)))
suffix = 'both'

# # Define the fake observations from right side only
# no_obs = 5  # Number of pairs of observations to be fed to the model
# # Indices of the actions for the margin map
# action_indices = np.array([1]*no_obs)
# margin_map = {0: -1, 1: 2.}
# observation_map = {0: 1., 1: 0.}
# fake_margins = np.array(
#     list(map(lambda x: margin_map[x], action_indices)))
# fake_observations = np.array(
#     list(map(lambda x: observation_map[x], action_indices)))
# suffix = 'right'


# Define parameters
no_sim = 20  # Number of simulations
T = 2000  # Number of time steps
no_actions = 129  # Number of actions
action_set = np.linspace(0, 0.9, no_actions, endpoint=True)

# Initialize the agents
tau_logistic = 40  # Window size for sliding window method
tau_classic = 40
epsilon = 0.05  # Epsilon for epsilon-greedy bandit
variance = 1.
dimension = 2

# Plotting parameters
results_folder = f'results/prior_me/logistic_enters_classic_no_actions_{no_actions}_T_{T}_{suffix}'


# Initialize the model
model_1 = NonStationaryClassicModel(
    variance=variance, candidate_margins=action_set, method='sliding_window', tau=tau_classic)
# Initialize epsilon-greedy bandit
bandit_1 = EpsGreedy(eps=epsilon, model=model_1)

# Initialize the model
model_2 = NonStationaryLogisticModel(
    dimension=dimension, candidate_margins=action_set, method='sliding_window', tau=tau_logistic)
# Feed fake observations to the model
model_2.set_prior_observations(fake_margins, fake_observations)
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
                       'Classic Eps-Greedy', 'Logistic Eps-Greedy', simulation_setup)
