"""This file tests the epsilon greedy bandit algorithm against
a dummy agent which always quotes the same fixed action."""

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

from models import NonStationaryLogisticModel

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
# np.random.seed(0)

# Define parameters
no_sim = 200  # Number of simulations
T = 1000  # Number of time steps
no_actions = 17  # Number of actions
action_set = np.linspace(-0.2, 0.9, no_actions, endpoint=True)  # Action set

epsilon = 0.05  # Epsilon for epsilon-greedy bandit
dimension = 2  # dimension of the model

fixed_action = 0.5  # Action for fixed-action agent

# Folder parameters
image_folder = f'results/logistic_upward_bias/epsgreedy_logistic_epsilon_{epsilon}'
ut.create_folder(image_folder)


# Initialize the environment
env = InsuranceMarketCt(**params['environment_parameters'])  # Environment

# Initialize the model
model = NonStationaryLogisticModel(
    candidate_margins=action_set, dimension=dimension, tau=40, method='sliding_window')
# Initialize epsilon-greedy bandit
bandit = EpsGreedy(eps=epsilon, model=model)
# Initialize dummy agent
fixed_agent = FixedActionAgent(action=fixed_action)


logger.info(f'Action set {action_set}')

# Run simulations
# reward_history, bandit_action_frequencies, fixed_action_frequencies = ut.run_simulations(
# bandit, fixed_agent, env, T, no_sim)
ut.single_game(bandit, fixed_agent, env, T)

expected_rewards_eps_005 = bandit.model.get_expected_rewards()
mean_rewards_eps_005 = bandit.model.get_mean_rewards()

bandit.reset()
bandit.eps = 1.
ut.single_game(bandit, fixed_agent, env, T)
# Get expected rewards for bandit with epsilon = 1
expected_rewards_eps_1 = bandit.model.get_expected_rewards()
mean_rewards_eps_1 = bandit.model.get_mean_rewards()


# Plot the expected reward plot for environment
_, _, _, mean_rewards, _ = ut.get_reward_profile(
    env, no_samples=2000, action_set=action_set, fixed_action=fixed_action)
plt.plot(action_set, mean_rewards, label='True expected reward')
plt.plot(action_set, expected_rewards_eps_005,
         label='Logistic bandit epsilon = 0.05 quantile 0.5')
plt.plot(action_set, mean_rewards_eps_005,
         label='Logistic bandit epsilon = 0.05 mean')
plt.plot(action_set, expected_rewards_eps_1,
         label='Logistic bandit epsilon = 1. quantile 0.5')
plt.plot(action_set, mean_rewards_eps_1,
         label='Logistic bandit epsilon = 1. mean')
plt.xlabel('Action')
plt.ylabel('Expected reward')
plt.legend()
plt.grid()
plt.savefig(os.path.join(image_folder, 'expected_reward_env.png'))
plt.show()
