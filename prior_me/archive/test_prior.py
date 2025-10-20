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
low_action = -2
action_set = np.array([low_action, high_action, 0.3, 0.5, 0.9])  # Action set

dimension = 2  # dimension of the model

model = StationaryLogisticModel(
    candidate_margins=action_set, dimension=dimension)


# Define the fake observations
no_obs = 5
action_indices = np.array([0, 1, 2, 3]*no_obs)
margin_map = {0: -4, 1: 4., 2: -2, 3: 2.}
observation_map = {0: 1., 1: 0., 2: 1., 3: 0.}
margins = np.array(
    list(map(lambda x: margin_map[x], action_indices)))
observations = np.array(
    list(map(lambda x: observation_map[x], action_indices)))


# Feed the observations to the model
model.set_prior_observations(margins, observations)

print(model.prior_xks)
print(model.prior_Ys)

# Compute mean and covariance
for i in range(10):
    mean, cov = model.compute_mean_and_cov()
    model.mean = mean
    model.cov = cov

# Print the mean and covariance of the model
logger.info("Mean: %s", mean)
logger.info("Covariance: %s", cov)
# exit(0)

lut.plot_logistic(model.mean.reshape(-1, 2), [
                  low_action, high_action], 'Prior', filename='single_mean.png')

means = np.random.multivariate_normal(model.mean.flatten(), model.cov, 10)
lut.plot_logistic(means, [low_action, high_action],
                  'Prior', filename='many_mean.png')

# exit(0)
# Print the mean and covariance of the model to a text file
# with open('prior_me/pessimistic_prior.txt', 'w') as f:
#     f.write(f'Mean: {model.mean}\n')
#     f.write(f'Covariance: {model.cov}\n')

# model.set_prior(prior_mean=model.mean, prior_cov=model.cov)
# model.update(3, None, 0.)
for i in range(3):
    model.update(3, None, 0.)

# Compute mean and covariance
# for i in range(10):
#     mean, cov = model.compute_mean_and_cov()
#     model.mean = mean
#     model.cov = cov

# Print the mean and covariance of the model
logger.info("Mean: %s", model.mean)
logger.info("Covariance: %s", model.cov)

lut.plot_logistic(model.mean.reshape(-1, 2), [
                  low_action, high_action], 'Prior after new observation',
                  filename='single_mean_after.png')

means = np.random.multivariate_normal(model.mean.flatten(), model.cov, 10)
lut.plot_logistic(means, [low_action, high_action],
                  'Prior after new observation', filename='many_mean_after.png')
