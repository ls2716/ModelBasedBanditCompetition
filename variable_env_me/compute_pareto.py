"""This script computes the Pareto equilibrium point given two players
and a number of actions."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import logging

import json
import utils as ut

from environment_analysis import environment_functions

# Set up logger function
logger = ut.get_logger(__name__)
logger.setLevel(logging.INFO)

# Load the environment parameters
# Read common parameters from yaml file
params = ut.read_parameters('variable_env_me/environment_parameters.yaml')
logger.info(json.dumps(params, indent=4))

env_params = params["environment_parameters_sigma_0.70"]

# Set seed for numpy
np.random.seed(0)


def compute_pareto(no_actions):
    """Compute the Pareto equilibrium points given two players and a number of actions."""
    action_set = np.linspace(0.2, 1.2, no_actions, endpoint=True)

    pareto_reward = 0
    rewards = []

    # Find th lowest nash point
    for S_i in action_set:
        true_probs, true_rewards = \
            environment_functions.expected_reward_probability(
                env_params=env_params,
                S_i=np.array([S_i]), S=[S_i], no_samples=40000
            )
        if true_rewards[0] > pareto_reward:
            pareto_reward = true_rewards[0]
            pareto_point = S_i
        rewards.append(true_rewards[0])

    return pareto_point, pareto_reward, rewards


if __name__ == "__main__":

    nums_of_actions = [501]

    for no_actions in nums_of_actions:
        pareto_point, pareto_reward, rewards = compute_pareto(no_actions)
        print(
            f'No actions {no_actions}, pareto point {pareto_point} with reward {pareto_reward}.')
