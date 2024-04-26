import sys

import numpy as np

__all__ = ['place_bayes']


def place_bayes(fr: np.ndarray,
                rate_map: np.ndarray,
                spatial_bin_size: float) -> np.ndarray:
    """
    Position decoding using population neuronal activity

    `Dimension parameters`:

        N = number of neurons

        T = number of temporal bins

        X = number of spatial bins


    .. seealso::

        https://github.com/buzsakilab/buzcode/blob/master/analysis/positionDecoding/placeBayes.m


    :param fr: firing rate 2D array. (T, N)
    :param rate_map: firing rate template. (X, N)
    :param spatial_bin_size: spatial bin size in cm

    :return: matrix of posterior probabilities. (T, X)
    """
    overflow = sys.float_info.min

    fr *= spatial_bin_size
    term2 = (-1) * spatial_bin_size * np.sum(rate_map, axis=1)
    term2 -= np.max(term2)  # normalize in log space
    term2 = np.exp(term2)  # (X,)

    if np.any(term2 < overflow):
        raise RuntimeError(f'OVERFLOW: {term2=}')

    outer_multiply = np.vectorize(np.multiply.outer, signature='(t),(s)->(t,s)')
    # outer_multiply((N, T), (N, X)) = (N, T, X)
    u = np.sum(outer_multiply(fr.T, np.log(rate_map.T)), axis=0)
    u -= np.max(u, axis=1, keepdims=True)  # normalize in log space
    u = np.exp(u)  # (T, X)

    if np.any(u < overflow):
        raise RuntimeError(f'OVERFLOW: {u=}')

    n_spatial_bins = rate_map.shape[0]
    pr = u * term2 / n_spatial_bins  # (T, X)

    pr /= np.sum(pr, axis=1, keepdims=True)

    if np.any(np.isnan(pr)):
        raise RuntimeError('pr approach the infinite')

    return pr
