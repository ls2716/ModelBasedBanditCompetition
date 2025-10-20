"""This file contains games of non-stationary agents between each other where one
agent is fully converged.

In this directory, the environment is different from the base environment.
Specifically, the noise level sigma can be varied between the runs
and it is set by setting env_name parameter to the desired value.
E.g env_name = 'sigma_1.00' sets the noise level to 1.00.

The nash and pareto payoffs are sourced from the environment_parameters file
according to the env_name. The payoffs are computed using other scripts
and they have to be manually input into the environment_parameters file.

This file corresponds to games where the classic agent enters the game
with the logistic agent who is already fully converged.

In this file, we plot the results from the simulations.
"""


# Import packages
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import os


logger = ut.get_logger(__name__)

# Set the environment name to source the parameters
env_name = 'sigma_1.00'

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

# Set seed
np.random.seed(0)


# Define parameters
T = 2000  # Number of time steps
no_actions = 129  # Number of actions
tau = 40  # Window size for sliding window method

# Plotting parameters
results_folder = f'results/variable_env_me/classic_enters_logistic_no_actions_{no_actions}_T_{T}_{env_name}'
images_folder = f'images/variable_env_me/classic_enters_logistic_no_actions_{no_actions}_T_{T}_{env_name}'

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
logger.info(f'Mean action of the bandit 1 (logistic): {bandit_1_mean_action}')
logger.info(f'Mean action of the bandit 2 (classic): {bandit_2_mean_action}')
logger.info(f'Mean reward of the bandit 1 (logistic): {reward_1_mean_500}')
logger.info(f'Mean reward of the bandit 2 (classic): {reward_2_mean_500}')
