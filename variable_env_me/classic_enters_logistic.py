"""This file contains games of non-stationary agents between each other where one
agent is fully converged.

In this directory, the environment is different from the base environment.
Specifically, the noise level sigma can be varied between the runs
and it is set by setting env_name parameter to the desired value.
E.g env_name = 'sigma_1.00' sets the noise level to 1.00.

The nash and pareto payoffs are sourced from the environment_parameters file
according to the env_name. The payoffs are computed using other scripts
and they have to be manually input into the environment_parameters file.

In this file specifically, we test the case where the classic agent enters the game
with the logistic agent who is already fully converged.
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

from variable_env_me import me_utils

logger = ut.get_logger(__name__)

# Set the environment name to source the parameters
env_name = 'sigma_0.10'

# Read the parameters from the yaml file
params = ut.read_parameters(f'variable_env_me/environment_parameters.yaml')

# Set the name of the environment to be used
full_env_name = f'environment_parameters_{env_name}'
logger.info(f'Using the environment {full_env_name}')
# Log the parameters
logger.info(json.dumps(params[full_env_name], indent=4))
# Source the nash and pareto payoffs from the file
nash_payoff = params[full_env_name]['nash_payoff']
pareto_payoff = params[full_env_name]['pareto_payoff']


def dimless_payoff(x):
    return (x - nash_payoff) / (pareto_payoff - nash_payoff)


# Set seed
np.random.seed(0)

# Initialize the environment
env = InsuranceMarketCt(**params[full_env_name])  # Environment

# Define parameters
no_sim = 20  # Number of simulations
T = 2000  # Number of time steps
no_actions = 129  # Number of actions
action_set = np.linspace(0, 0.9, no_actions, endpoint=True)


# Initialize the agents
tau_logistic = 40  # Sliding window for logistic agent
tau_classic = 40  # Sliding window for classic agent
epsilon = 0.05  # Epsilon for epsilon-greedy bandit
variance = 1.
dimension = 2

# Plotting parameters
results_folder = f'results/variable_env_me/classic_enters_logistic_no_actions_{no_actions}_T_{T}_{env_name}'


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
                       'Logistic Eps-Greedy', 'Classic Eps-Greedy', simulation_setup,
                       dimless_payoff=dimless_payoff, env=env)
