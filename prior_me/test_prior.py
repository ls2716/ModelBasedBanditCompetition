"""Computing the pessimistic prior for the logistic model

This is done by prefeeding the logistic model with some specified
fake observations which express the prior belief of the environment.

In this case this will correspond to probability one of getting the order 
if the margin is small negative (e.g. -0.5) and probability zero if the margin
is large positive (e.g. 2.).
"""

from models import StationaryLogisticModel
import utils as ut
import prior_me.logistic_utils as lut

# Import packages
import numpy as np
import logging
import os
import json

# Set up the logger
logger = ut.get_logger(__name__)

# Specify actions
no_actions = 129  # Number of actions
action_set = np.linspace(0, 0.9, no_actions, endpoint=True)
dimension = 2  # dimension of the model

model = StationaryLogisticModel(
    candidate_margins=action_set, dimension=dimension)


# Define the fake observations for both sides
no_obs = 5  # Number of pairs of observations to be fed to the model
# Indices of the actions for the margin map
action_indices = np.array([0, 1]*no_obs)
margin_map = {0: -1, 1: 2.}
observation_map = {0: 1., 1: 0.}
margins = np.array(
    list(map(lambda x: margin_map[x], action_indices)))
observations = np.array(
    list(map(lambda x: observation_map[x], action_indices)))
suffix = 'both'

# # Define the fake observations for right side only
# no_obs = 5  # Number of pairs of observations to be fed to the model
# # Indices of the actions for the margin map
# action_indices = np.array([1]*no_obs)
# margin_map = {0: -1, 1: 2.}
# observation_map = {0: 1., 1: 0.}
# margins = np.array(
#     list(map(lambda x: margin_map[x], action_indices)))
# observations = np.array(
#     list(map(lambda x: observation_map[x], action_indices)))
# suffix = 'right'

# # Then:
# # margins = [-2, 2] repeated no_obs times
# # observations = [1, 0] repeated no
# print(margins)
# print(observations)
# suffix = 'none'


# Feed the observations to the model
model.set_prior_observations(margins, observations)

print(model.prior_xks)
print(model.prior_Ys)

# Compute mean and covariance
for i in range(10):
    mean, cov = model.compute_mean_and_cov()
    model.mean = mean
    model.cov = cov

# # Print the mean and covariance of the model
# logger.info("Mean: %s", mean)
# logger.info("Covariance: %s", cov)

# Plotting
low_action = -1.
high_action = 2.

images_folder = f'images/prior_me/{suffix}_pessimistic_prior'
ut.create_folder(images_folder)

lut.plot_logistic(model.mean.reshape(-1, 2), [
    low_action, high_action], 'Prior', filename=os.path.join(images_folder, 'single_mean.png'))

means = np.random.multivariate_normal(model.mean.flatten(), model.cov, 50)
lut.plot_logistic(means, [low_action, high_action],
                  'Prior', filename=os.path.join(images_folder, f'{suffix}_many_mean.png'))
exit(0)

new_action_index = 60
new_observation = 0.
print(
    f'Feeding new observations at action index {new_action_index} with observation {new_observation}')
print(f'Corresponding margin {action_set[new_action_index]}')
for i in range(3):
    model.update(60, None, 0.)


# Print the mean and covariance of the model
logger.info("Mean: %s", model.mean)
logger.info("Covariance: %s", model.cov)

lut.plot_logistic(model.mean.reshape(-1, 2), [
                  low_action, high_action], 'Prior after new observation',
                  filename=os.path.join(images_folder, 'single_mean_after.png'))

means = np.random.multivariate_normal(model.mean.flatten(), model.cov, 10)
lut.plot_logistic(means, [low_action, high_action],
                  'Prior after new observation', filename=os.path.join(images_folder, 'many_mean_after.png'))
