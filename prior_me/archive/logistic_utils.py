import numpy as np
import matplotlib.pyplot as plt
import os

from models import sigmoid


def plot_logistic(means, x_lims, title, filename, show_plot=True):
    """Plot logistic function for given mean."""
    x = np.linspace(x_lims[0], x_lims[1], 100)
    for i in range(means.shape[0]):
        y = sigmoid(means[i][0] + means[i][1]*x)
        plt.plot(x, y, label=f'Mean {means[i]}')
    plt.xlabel('Margin')
    plt.ylabel('Probability')
    plt.legend()
    plt.title(title)
    plt.savefig(filename)
    if show_plot:
        plt.show()
