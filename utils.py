import numpy as np
import matplotlib.pyplot as plt


def plot_data(X, y, ax):
    """
    Plots the data points for logistic regression.

    Args:
        X (ndarray): Input features with shape (m, 2)
        y (ndarray): Target values with shape (m, 1)
        ax: Matplotlib axis
    """

    # Plot positive examples
    pos = y.flatten() == 1
    ax.scatter(
        X[pos, 0],
        X[pos, 1],
        marker='x',
        s=80,
        linewidth=2,
        label='y = 1'
    )

    # Plot negative examples
    neg = y.flatten() == 0
    ax.scatter(
        X[neg, 0],
        X[neg, 1],
        marker='o',
        s=80,
        label='y = 0'
    )

    ax.legend()