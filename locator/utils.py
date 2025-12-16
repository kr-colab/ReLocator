"""Utility functions for data processing"""

import numpy as np

__all__ = [
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


# Legacy imports for backward compatibility
# These are now defined in sample_weights.py but we keep them available here

# Import weight_samples from the dedicated module
from .sample_weights import weight_samples
