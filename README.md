# Learning Extortion in Pricing Duopoly Competition


This repository contains the code accompanying the paper:  
“Competitive Pricing Using Model-Based Bandits”  
published in the *Computational Economics* (2025).

The code supports the research presented in Chapter 2, "Competitive Pricing using
Model-based Bandits" of the PhD thesis:  
“Algorithmic Pricing in Competitive Markets”  
by Lukasz Sliwinski, 2025, University of Edinburgh.

---

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

The scripts should be run using the module flag `-m` from the repository root.

Each folder corresponds to a section of the paper and contains scripts to reproduce the results. Below is the list. The scripts not included are utility scripts which define classes and functions used by the main scripts.


### environment_analysis

Environment analysis scripts.

 - reward_plot.py - produces a reward plot given an environment configuration.
 - best_response_plot.py - produces a best response plot given an environment configuration and opponent action.
 - derivative_analysis.py - analyses the derivates of the best response function (appendix checking the environment assumptions).

### model_error

Scripts which analysis the error between the logistic and true demand functions.

 - analyze_error_sample_params.py - analyzes the model error for a sample set of environment parameters.
 - analyze_error.py - analyzes the model error over a range of environment parameters.
 - compute_nash.py - computes the Nash equilibrium for given environment parameters.
 - compute_pareto.py - computes the symmetric Pareto equilibrium for given environment parameters.

### fixed-action_agent

Contains the experiments of learning agents against a stationary fixed-action agent.

- epsgreedy_classic.py - runs the epsilon-greedy classic agent experiment.
- epsgreedy_classic_plot.py - plots the results of the epsilon-greedy classic agent experiment.
- epsgreedy_logistic.py - runs the epsilon-greedy logistic agent experiment.
- epsgreedy_logistic_plot.py - plots the results of the epsilon-greedy logistic agent experiment.
- ucb_classic.py - runs the UCB classic agent experiment.
- ucb_classic_plot.py - plots the results of the UCB classic agent experiment.
- ucb_logistic.py - runs the UCB logistic agent experiment.
- ucb_logistic_plot.py - plots the results of the UCB logistic agent experiment.

### fluctuating_agent

Contains the experiments of learning agents against an opponent with fluctuating price.

- epsgreedy_classic.py - runs the epsilon-greedy classic agent experiment.
- epsgreedy_classic_plot.py - plots the results of the epsilon-greedy classic agent experiment.
- epsgreedy_logistic.py - runs the epsilon-greedy logistic agent experiment.
- epsgreedy_logistic_plot.py - plots the results of the epsilon-greedy logistic agent experiment.
- ucb_classic.py - runs the UCB classic agent experiment.
- ucb_classic_plot.py - plots the results of the UCB classic agent experiment.
- ucb_logistic.py - runs the UCB logistic agent experiment.
- ucb_logistic_plot.py - plots the results of the UCB logistic agent experiment.

### market_entering

Contains the experiments of competitions where one agent enters the market after the other agent has already been present for some time.

- classic_enters_classic.py - runs the classic agent entering a market with a classic agent experiment.
- classic_enters_classic_plot.py - plots the results of the classic agent entering a market with a classic agent experiment.
- classic_enters_logistic.py - runs the classic agent entering a market with a logistic agent experiment.
- classic_enters_logistic_plot.py - plots the results of the classic agent entering a market with a logistic agent experiment.
- logistic_enters_classic.py - runs the logistic agent entering a market with a classic agent experiment.
- logistic_enters_classic_plot.py - plots the results of the logistic agent entering a market with a classic agent experiment.
- logistic_enters_logistic.py - runs the logistic agent entering a market with a logistic agent experiment.
- logistic_enters_logistic_plot.py - plots the results of the logistic agent entering a market with a logistic agent experiment.

### market_entering_extended

Contains the same experiments as in `market_entering` but with longer episodes times to compute the long-term convergence.

- classic_enters_logistic.py - runs the classic agent entering a market with a logistic agent experiment.
- classic_enters_logistic_plot.py - plots the results of the classic agent entering a market with a logistic agent experiment.
- logistic_enters_classic.py - runs the logistic agent entering a market with a classic agent experiment.
- logistic_enters_classic_plot.py - plots the results of the logistic agent entering a market with a classic agent experiment.

### variable_env_me

Repeats the above experiments but with different environment parameters to analyze the impact of the environment on the results.
- classic_enters_logistic.py - runs the classic agent entering a market with a logistic agent experiment.
- classic_enters_logistic_plot.py - plots the results of the classic agent entering a market with a logistic agent experiment.
- logistic_enters_classic.py - runs the logistic agent entering a market with a classic agent experiment.
- logistic_enters_classic_plot.py - plots the results of the logistic agent entering a market with a classic agent experiment.

### prior_me

Contains experiments analyzing the impact of prior information on the performance of the logistic agent.

-test_priors.py - Analyses the prior information that can be fed to the logistic agent. The script adds artificial observations to ensure that the modeled demand curve is decreasing.
- classic_enters_logistic.py - runs the classic agent entering a market with a logistic agent experiment with different priors.
- classic_enters_logistic_plot.py - plots the results of the classic agent entering a market with a logistic agent experiment with different priors.
- logistic_enters_classic.py - runs the logistic agent entering a market with a classic agent experiment with different priors.
- logistic_enters_classic_plot.py - plots the results of the logistic agent entering a market with a classic agent experiment with different priors.

## Additional files

- bandits.py - contains the implementation of the learning agents.  
- envs.py - contains the implementation of the pricing duopoly environment.
- models.py - contains the implementation of the environment models used by the agents (logistic/classic; stationary/non-stationary).
- utils.py - contains utility functions used across the scripts.

