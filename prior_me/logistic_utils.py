import numpy as np
import matplotlib.pyplot as plt
import os

from models import sigmoid

# Set the font size
plt.rcParams.update({'font.size': 19})


def plot_logistic(means, x_lims, title, filename, show_plot=True):
    """Plot logistic function for given mean."""
    x = np.linspace(x_lims[0], x_lims[1], 100)
    plt.figure(figsize=(6.5, 5))
    for i in range(means.shape[0]):
        y = sigmoid(means[i][0] + means[i][1]*x)
        plt.plot(x, y, label=f'Mean {means[i]}')
    plt.xlabel('Margin')
    plt.ylabel('Probability')
    # plt.legend()
    # plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    if show_plot:
        plt.show()
