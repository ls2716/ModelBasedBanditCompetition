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
no_actions = 2  # Number of actions
high_action = 2.
low_action = -0.5
action_set = np.array([low_action, high_action, 0])  # Action set

dimension = 2  # dimension of the model

model = StationaryLogisticModel(
    candidate_margins=action_set, dimension=dimension)


# Define the fake observations
no_obs = 30
action_indices = np.array([0, 1]*no_obs)
observation_map = {0: 1., 1: 0.}

# Feed the observations to the model
for i in range(len(action_indices)):
    action = action_indices[i]
    logger.info("Action: %s", action)
    observation = observation_map[action_indices[i]]
    model.update(action, None, observation)

# Print the mean and covariance of the model
logger.info("Mean: %s", model.mean)
logger.info("Covariance: %s", model.cov)

lut.plot_logistic(model.mean.reshape(-1, 2), [
                  low_action, high_action], 'Pessimistic prior')

means = np.random.multivariate_normal(model.mean.flatten(), model.cov, 10)
lut.plot_logistic(means, [low_action, high_action], 'Pessimistic prior')

# Print the mean and covariance of the model to a text file
with open('prior_me/pessimistic_prior.txt', 'w') as f:
    f.write(f'Mean: {model.mean}\n')
    f.write(f'Covariance: {model.cov}\n')

# model.set_prior(prior_mean=model.mean, prior_cov=model.cov)
for i in range(10):
    model.update(2, None, 0.)

# Print the mean and covariance of the model
logger.info("Mean: %s", model.mean)
logger.info("Covariance: %s", model.cov)

lut.plot_logistic(model.mean.reshape(-1, 2), [
                  low_action, high_action], 'Pessimistic prior after new observation')

means = np.random.multivariate_normal(model.mean.flatten(), model.cov, 10)
lut.plot_logistic(means, [low_action, high_action], 'Pessimistic prior')
