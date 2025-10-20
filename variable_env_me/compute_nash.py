"""This script computes the Nash equilibrium point given two players
and a number of actions.

This script uses the parameters from the environment_parameters.yaml file
in that file, there are multiple environment for different noise levels.

"""

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

# Increase the font size of the plots
plt.rcParams.update({'font.size': 15})

# Load the environment parameters
# Read common parameters from yaml file
params = ut.read_parameters('variable_env_me/environment_parameters.yaml')
logger.info(json.dumps(params, indent=4))

env_name = 'sigma_1.00'
env_params = params[f"environment_parameters_{env_name}"]

# Set seed for numpy
np.random.seed(0)

# Set the image folder
image_folder = 'images/variable_env_me/nash_reward_probability'


# Create the images folder
ut.create_folder(image_folder)


def compute_nash(no_actions):
    """Compute the Nash equilibrium points given two players and a number of actions."""
    action_set = np.linspace(0., 0.9, no_actions, endpoint=True)

    nash_points = []

    # Find th lowest nash point
    S_0 = action_set[0]
    while True:
        true_probs, true_rewards = \
            environment_functions.expected_reward_probability(
                env_params=env_params,
                S_i=action_set, S=[S_0], no_samples=40000
            )
        s_0_index = np.argmax(true_rewards)
        new_S_0 = action_set[s_0_index]

        if S_0 == new_S_0:
            nash_points.append((s_0_index, new_S_0, true_rewards[s_0_index]))
            break
        S_0 = new_S_0

    for it in range(s_0_index+1, no_actions):
        action = action_set[it]
        true_probs, true_rewards = \
            environment_functions.expected_reward_probability(
                env_params=env_params,
                S_i=action_set, S=[action], no_samples=40000
            )
        index = np.argmax(true_rewards)
        new_S_0 = action_set[index]
        # print(f'Best against {S_0} is {new_S_0}')
        if action == new_S_0:
            nash_points.append((index, action, true_rewards[index]))
        elif len(nash_points) > 0:
            break

    # Plot the rewards for the last action
    plt.plot(action_set, true_rewards)
    plt.xlabel('Margin')
    plt.ylabel('Reward')
    # plt.title('Reward at the Nash equilibrium')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{image_folder}/rewards_{env_name}.pdf')
    plt.savefig(f'{image_folder}/rewards_{env_name}.png', dpi=300)
    plt.show()
    # Plot the probabilities for the last action
    plt.rcParams.update({'font.size': 19})
    plt.figure(figsize=(6.5, 5))
    plt.plot(action_set, true_probs)
    plt.xlabel('Margin')
    plt.ylabel('Probability')
    plt.ylim(-0.2, 1.2)
    # plt.title('Probability at the Nash equilibrium')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{image_folder}/probabilities_{env_name}.pdf')
    plt.savefig(f'{image_folder}/probabilities_{env_name}.png', dpi=300)
    plt.show()

    return nash_points


if __name__ == "__main__":

    nums_of_actions = [501]

    for no_actions in nums_of_actions:
        nash_points = compute_nash(no_actions)
        print(
            f'No actions {no_actions}, nash_points {nash_points}')
