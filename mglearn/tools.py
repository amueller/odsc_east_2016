import numpy as np
from sklearn.datasets import make_blobs

from .plot_2d_separator import plot_2d_separator, plot_2d_classification, plot_2d_scores
from .plot_helpers import cm2 as cm


def make_handcrafted_dataset():
    # a carefully hand-designed dataset lol
    X, y = make_blobs(centers=2, random_state=4, n_samples=30)
    y[np.array([7, 27])] = 0
    mask = np.ones(len(X), dtype=np.bool)
    mask[np.array([0, 1, 5, 26])] = 0
    X, y = X[mask], y[mask]
    return X, y

__all__ = ['plot_2d_separator', 'plot_2d_classification',
           'plot_2d_scores', 'cm']
