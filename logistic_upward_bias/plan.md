# Upward logistic bias

In this directory, we perform experiments to verify that the upward bias observed in a
pricing competition with a logistic agent is a result of a bias of information and not an
inherent bias of the model.

This will thus include following steps:

1. Simulate a competition between fixed agent and classic logistic agent with low
   epsilon - this reflects the bias when the agent is fed biased information.
2. Simulate a competition between a fixed agent and a classic logistic agent with a high
   value of epsilon - this simulates a scenario where there is no bias of information.
3. For both these scenarios, plot the expected reward for each action and juxtapose it
   against the true reward curve.
4. The plots are the output of the simulation - use them to draw conclusions.
