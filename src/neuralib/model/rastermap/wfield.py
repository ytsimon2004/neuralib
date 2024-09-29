from typing import NamedTuple

import numpy as np
from sklearn.decomposition import TruncatedSVD

__all__ = ['SequenceSingularVector',
           'compute_singular_vector']


class SequenceSingularVector(NamedTuple):
    """A NamedTuple that represents a singular value decomposition result from image sequences."""

    singular_value: np.ndarray
    """The singular values of the decomposition"""
    right_vector: np.ndarray
    """Right singular vectors of the decomposition"""
    left_vector: np.ndarray
    """Left singular vectors of the decomposition"""


def compute_singular_vector(sequences: np.ndarray,
                            n_components: int = 128) -> SequenceSingularVector:
    """
    :param sequences: A numpy array representing a collection of image frames in a sequence.
                      It has a shape of (n_frames, width, height) where 'n_frames' is the number
                      of frames, and 'width' and 'height' are the dimensions of each frame.
    :param n_components: An integer representing the number of components for Truncated SVD.
                         The default value is 128.
    :return: A SequenceSingularVector object containing the singular values, the transformed
             components, and the left singular vectors.
    """
    n_frames, width, height = sequences.shape
    seq = sequences.reshape(n_frames, height * width).T
    seq = seq - np.mean(seq, axis=1, keepdims=True)

    svd = TruncatedSVD(n_components=n_components)
    Vsv = svd.fit_transform(seq.T)
    U = seq @ (Vsv / (Vsv ** 2).sum(axis=0) ** 0.5)
    U /= (U ** 2).sum(axis=0) ** 0.5

    return SequenceSingularVector(svd.singular_values_, Vsv, U)
