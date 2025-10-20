"""This file tests the epsilon greedy bandit algorithm against
a dummy agent which always quotes the same fixed action.

This time with prior.
"""

# Import packages
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import os

# Import bandit algorithms
from bandits import EpsGreedy

# Import fixed action agent
from bandits import FixedActionAgent

from models import StationaryLogisticModel

# Initialize the environment
from envs import InsuranceMarketCt


logger = ut.get_logger(__name__)

# Read common parameters from yaml file
params = ut.read_parameters('common_parameters.yaml')
logger.info(json.dumps(params, indent=4))


# Define dimless payoff function
nash_payoff = params['environment_parameters']['nash_payoff']
pareto_payoff = params['environment_parameters']['pareto_payoff']


def dimless_payoff(x):
    return (x - nash_payoff) / (pareto_payoff - nash_payoff)


# Set seed
np.random.seed(0)

# Define parameters
no_sim = 20  # Number of simulations
T = 1000  # Number of time steps
no_actions = 5  # Number of actions
action_set = np.linspace(0.1, 0.9, no_actions, endpoint=True)  # Action set

epsilon = 0.05  # Epsilon for epsilon-greedy bandit
dimension = 2  # dimension of the model

fixed_action = 0.7  # Action for fixed-action agent

# Folder parameters
result_folder = f'results/prior_me/epsgreedy_logistic_{no_actions}_obs'


# Initialize the environment
env = InsuranceMarketCt(**params['environment_parameters'])  # Environment

# Initialize the model
model = StationaryLogisticModel(
    candidate_margins=action_set, dimension=dimension)

# prior_mean = np.array([[8.03620972e-04], [-3.56847822e+00]])
# prior_cov = np.array([[2.36311068e+01, -1.00809722e-02],
#                      [-1.00809722e-02,  6.01437130e+00]])

# Set prior for the logistic model
# model.set_prior(prior_mean=prior_mean, prior_cov=prior_cov)
# Define the fake observations
no_obs = 20
action_indices = np.array([0, 1, 2, 3]*no_obs)
margin_map = {0: -4, 1: 4., 2: -2, 3: 2.}
observation_map = {0: 1., 1: 0., 2: 1., 3: 0.}
# action_indices = np.append(action_indices, [2, 3])
margins = np.array(
    list(map(lambda x: margin_map[x], action_indices)))
observations = np.array(
    list(map(lambda x: observation_map[x], action_indices)))
# Feed the observations to the model
model.set_prior_observations(margins, observations)

# Initialize epsilon-greedy bandit
bandit = EpsGreedy(eps=epsilon, model=model)
# Initialize dummy agent
fixed_agent = FixedActionAgent(action=fixed_action)

logger.info(f'Bandit prior mean {bandit.model.mean}')
logger.info(f'Bandit prior cov {bandit.model.cov}')


logger.info(f'Action set {action_set}')
# exit(0)
# Run simulations
reward_history, bandit_action_frequencies, fixed_action_frequencies = ut.run_simulations(
    bandit, fixed_agent, env, T, no_sim)


reward_history = dimless_payoff(reward_history)

# Print cumulative reward for the bandit and save to a file
logger.info(f'Sum of rewards for the bandit: {np.sum(reward_history[:, 0])}')
reward_1_sum = np.sum(reward_history[:, 0])
reward_2_sum = np.sum(reward_history[:, 1])
ut.create_folder(result_folder)
ut.print_result_to_file(reward_1_sum, reward_2_sum, os.path.join(
    result_folder, 'total_rewards.txt'), 'Logistic Eps-Greedy', 'Fixed-Action Agent')

# Save action set to a csv file
np.savetxt(os.path.join(result_folder, 'action_set.csv'),
           action_set, delimiter=',')

# Save reward history to a csv file
np.savetxt(os.path.join(result_folder, 'reward_history.csv'),
           reward_history, delimiter=',')

# Save action frequencies to a csv file
np.savetxt(os.path.join(result_folder, 'bandit_action_frequencies.csv'),
           bandit_action_frequencies, delimiter=',')
