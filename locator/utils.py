"""Utility functions for data processing"""

import numpy as np, pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

__all__ = [
    "load_genotypes",
    "sort_samples",
    "weight_samples",
    "split_train_test",
]


def split_train_test(ac, locs, train_split=0.8):
    """Split data into training and test sets

    Args:
        ac: allele counts array
        locs: locations array
        train_split: proportion of data to use for training (default: 0.8)
    """
    train = np.argwhere(~np.isnan(locs[:, 0]))
    train = np.array([x[0] for x in train])
    pred = np.array([x for x in range(len(locs)) if x not in train])
    test = np.random.choice(train, round((1 - train_split) * len(train)), replace=False)
    train = np.array([x for x in train if x not in test])
    traingen = np.transpose(ac[:, train])
    trainlocs = locs[train]
    testgen = np.transpose(ac[:, test])
    testlocs = locs[test]
    predgen = np.transpose(ac[:, pred])
    return train, test, traingen, testgen, trainlocs, testlocs, pred, predgen

# Import weight_samples from the dedicated module
from .sample_weights import weight_samples


# Legacy imports for backward compatibility
# These are now defined in sample_weights.py but we keep them available here
from .sample_weights import _make_kd_weights, _make_histogram_weights, _load_sample_weights