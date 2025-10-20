"""This file contains games of non-stationary agents between each other where one
agent is fully converged.


These simulations are extended versions of the simulations and include variable 
logistic sliding window tau.

This file is used for plotting results and writing down the performance for
the simulations where the classic agent enters the environment in which 
the logistic agent is already present and fully converged.

The plotting is based on the simulations run
in market_entering_extended/classic_enters_logistic.py
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
from models import NonStationaryClassicModel

# Initialize the environment
from envs import InsuranceMarketCt

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
T = 2000  # Number of time steps
no_actions = 129  # Number of actions
tau = 150  # Window size for sliding window method

# Plotting parameters
results_folder = f'results/market_entering_extended/classic_enters_logistic_no_actions_{no_actions}_T_{T}_tau_{tau}'
images_folder = f'images/market_entering_extended/classic_enters_logistic_no_actions_{no_actions}_T_{T}_tau_{tau}'

# Create the images folder
ut.create_folder(images_folder)

# Load the data from the results folder
action_set = np.loadtxt(os.path.join(results_folder, 'action_set.csv'),
                        delimiter=',')
reward_history = np.loadtxt(os.path.join(results_folder, 'reward_history.csv'),
                            delimiter=',')
bandit_1_action_frequencies = np.loadtxt(os.path.join(results_folder, 'bandit_1_action_frequencies.csv'),
                                         delimiter=',')
bandit_2_action_frequencies = np.loadtxt(os.path.join(results_folder, 'bandit_2_action_frequencies.csv'),
                                         delimiter=',')


# Print cumulative reward for the bandit and save to a file
logger.info(f'Sum of rewards for the bandit: {np.sum(reward_history[:, 0])}')
reward_1_sum = np.sum(reward_history[:1000, 0])
reward_2_sum = np.sum(reward_history[:1000, 1])
ut.print_result_to_file(reward_1_sum, reward_2_sum, os.path.join(
    results_folder, 'first_1000.txt'), 'Logistic Eps-Greedy', 'Classic Eps-Greedy entering')

logger.info(
    f'Sum of last 500 rewards for the bandits:'
    + f' {np.sum(reward_history[-500:, 0])}, {np.sum(reward_history[-500:, 1])}')
reward_1_sum_500 = np.sum(reward_history[-500:, 0])
reward_2_sum_500 = np.sum(reward_history[-500:, 1])
ut.print_result_to_file(reward_1_sum_500, reward_2_sum_500, os.path.join(
    results_folder, 'last_500.txt'), 'Logistic Eps-Greedy', 'Classic Eps-Greedy entering')

logger.info(
    f'Mean of last 500 rewards for the bandits:'
    + f' {np.mean(reward_history[-500:, 0])}, {np.mean(reward_history[-500:, 1])}')
reward_1_mean_500 = np.mean(reward_history[-500:, 0])
reward_2_mean_500 = np.mean(reward_history[-500:, 1])
ut.print_result_to_file(reward_1_mean_500, reward_2_mean_500, os.path.join(
    results_folder, 'mean_last_500.txt'), 'Logistic Eps-Greedy', 'Classic Eps-Greedy entering')


ut.plot_smooth_reward_history(
    reward_history, bandit1_name='Logistic Eps-Greedy',
    bandit2_name='Classic Eps-Greedy entering', foldername=images_folder, filename='', title=None, show_plot=False)


# Get the mean action from the last 500 time steps
bandit_1_mean_action = action_set[:,
                                  None].T @ bandit_1_action_frequencies[-500:, :].T
bandit_1_mean_action = np.mean(bandit_1_mean_action)
bandit_2_mean_action = action_set[:,
                                  None].T @ bandit_2_action_frequencies[-500:, :].T
bandit_2_mean_action = np.mean(bandit_2_mean_action)


# Print the mean action to the logger
logger.info(f'Mean action of the bandit 1: {bandit_1_mean_action}')
logger.info(f'Mean action of the bandit 2: {bandit_2_mean_action}')
