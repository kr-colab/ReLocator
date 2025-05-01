"""Utility functions for data processing"""

import numpy as np, pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

__all__ = [
    "load_genotypes",
    "sort_samples",
    "normalize_locs",
    "filter_snps",
    "weight_samples",
]


def normalize_locs(locs):
    """Normalize location coordinates"""
    unnormedlocs = locs.copy()
    meanlong = np.nanmean(locs[:, 0])
    sdlong = np.nanstd(locs[:, 0])
    meanlat = np.nanmean(locs[:, 1])
    sdlat = np.nanstd(locs[:, 1])
    locs = np.array(
        [[(x[0] - meanlong) / sdlong, (x[1] - meanlat) / sdlat] for x in locs]
    )
    return meanlong, sdlong, meanlat, sdlat, unnormedlocs, locs


def replace_md(genotypes):
    """Replace missing data with binomial draws from allele frequency"""
    print("imputing missing data")
    dc = genotypes.count_alleles()[:, 1]
    ac = genotypes.to_allele_counts()[:, :, 1]
    missingness = genotypes.is_missing()
    ninds = np.array([np.sum(x) for x in ~missingness])
    af = np.array([dc[x] / (2 * ninds[x]) for x in range(len(ninds))])
    for i in tqdm(range(np.shape(ac)[0])):
        for j in range(np.shape(ac)[1]):
            if missingness[i, j]:
                ac[i, j] = np.random.binomial(2, af[i])
    return ac


def filter_snps(genotypes, min_mac=1, max_snps=None, impute=False, verbose=False):
    """Filter SNPs based on criteria.

    Args:
        genotypes: GenotypeArray to filter
        min_mac (int): Minimum minor allele count for filtering
        max_snps (int, optional): Maximum number of SNPs to retain
        impute (bool): Whether to impute missing data
        verbose (bool): Whether to print progress messages. Defaults to True.

    Returns:
        numpy.ndarray: Filtered allele counts array
    """
    if verbose:
        print("filtering SNPs")

    tmp = genotypes.count_alleles()
    biallel = tmp.is_biallelic()
    genotypes = genotypes[biallel, :, :]

    if min_mac > 1:
        derived_counts = genotypes.count_alleles()[:, 1]
        ac_filter = [x >= min_mac for x in derived_counts]
        genotypes = genotypes[ac_filter, :, :]

    if impute:
        ac = replace_md(genotypes)
    else:
        ac = genotypes.to_allele_counts()[:, :, 1]

    if max_snps is not None:
        ac = ac[np.random.choice(range(ac.shape[0]), max_snps, replace=False), :]

    if verbose:
        print(f"filtered {ac.shape[1]} individual genotypes")
        print(f"{ac.shape[0]} SNPs after filtering\n\n\n")

    return ac


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

def weight_samples(method,
                   trainlocs=None,
                   trainsamps=None,
                   weightdf=None,
                   xbins=None,
                   ybins=None,
                   lam=None,
                   bandwidth=None):
    """
    Calculate weights for training data based on the specified method
    Args:
        method (str): Method for calculating weights ('KD', 'histogram', or 'load')
        trainlocs (numpy.ndarray): Training locations
        weightdf (pd.DataFrame): DataFrame containing sample weights (default: None)
        xbins (int): Number of bins in x direction (default: 10)
        ybins (int): Number of bins in y direction (default: 10)
        lam (float): Exponent for weights (default: 1.0)
        bandwidth (float): Bandwidth for KDE (default: None)
    Returns:
        numpy.ndarray: Weights for training data
    """

    if method == 'KD':
        weights = _make_kd_weights(trainlocs, 
                                  1.0 if lam is None else lam, 
                                  bandwidth)
        df = pd.DataFrame({'sampleID':trainsamps,
                                      'sample_weight':weights})
    elif method == 'histogram':
        weights = _make_histogram_weights(trainlocs, 
                                      10 if xbins is None else xbins,
                                      10 if ybins is None else ybins)
        df = pd.DataFrame({'sampleID':trainsamps,
                                      'sample_weight':weights})
    elif method == 'load':
        df = _load_sample_weights(weightdf, trainsamps)
        weights = df['sample_weight'].values

    else:
        raise ValueError("Invalid method. Choose 'kde', 'histogram', or 'load'.")
    return {'method': method,
            'sample_weights': weights,
            'sample_weights_df': df,
            'xbins': xbins,
            'ybins': ybins,
            'lam': lam,
            'bandwidth': bandwidth,
            }


def _make_kd_weights(trainlocs, lam=1.0, bandwidth=None):
    """
    Calculate weights for training data using Kernel Density Estimation (KDE)
    Args:
        trainlocs (numpy.ndarray): Training locations
        lam (float): Exponent for weights (default: 1.0)
        bandwidth (float): Bandwidth for KDE (default: 
            GridSearchCV to find optimal bandwidth)
    Returns:
        numpy.ndarray: Weights for training data
    """
    if bandwidth:
        bw = bandwidth
    else:
    # use gridsearch to ID best bandwidth size
        bandwidths = np.linspace(0.1, 10, 1000)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth':bandwidths})
        grid.fit(trainlocs)
        bw = grid.best_params_['bandwidth']
    
    # fit kernel
    kde = KernelDensity(bandwidth=bw, kernel='gaussian')
    kde.fit(trainlocs)

    # calculate weights
    weights = kde.score_samples(trainlocs)
    weights = 1.0 / np.exp(weights)
    weights /= min(weights)

    weights = np.power(weights, lam)

    weights /= sum(weights)

    return weights

def _make_histogram_weights(trainlocs, xbins=10, ybins=10):
    """
    Calculate weights for training data using histogram binning
    Args:
        trainlocs (numpy.ndarray): Training locations
        xbins (int): Number of bins in x direction (default: 10)
        ybins (int): Number of bins in y direction (default: 10)
    Returns:
        numpy.ndarray: Weights for training data
    """
    bincount = [xbins, ybins]
    # make 2D histogram
    H, xedges, yedges = np.histogram2d(trainlocs[:,0], trainlocs[:, 1], bins=bincount)
    # sort trainlocs into bins
    xbin = np.digitize(trainlocs[:, 0], xedges[1:], right=True)
    ybin = np.digitize(trainlocs[:, 1], yedges[1:], right=True)
    # assign sample weights
    weights = np.empty(len(trainlocs), dtype='float')
    for i in range(len(trainlocs)):
        weights[i] = 1/(H[xbin[i]][ybin[i]])
    weights /= min(weights)

    return weights

def _load_sample_weights(weightdf, trainsamps):
    """Load sample weights from a DataFrame
    Args:
        weightdf (pd.DataFrame): DataFrame containing sample weights
        trainsamps (list): List of training sample IDs
    Returns:    
        numpy.ndarray: Array of sample weights
    """
    weightdf.set_index('sampleID', inplace=True)
    weights = np.empty(len(trainsamps), dtype='float')
    for i in range(len(trainsamps)):
        w = weightdf.loc[trainsamps[i], 'sample_weight']
        if type(w) == pd.core.series.Series:
            weights[i] = w[0]
        else:
            weights[i] = w 
    return np.array(weights)