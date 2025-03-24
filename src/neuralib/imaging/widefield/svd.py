from typing import NamedTuple

import numpy as np
from sklearn.decomposition import TruncatedSVD

__all__ = [
    'SequenceSingularVector',
    'compute_singular_vector'
]


class SequenceSingularVector(NamedTuple):
    """A NamedTuple that represents a singular value decomposition result from image sequences

    `Dimension parameters`:

        W = image width

        H = image height

        F = number of image frame (sequence)

        C = number of components after SVD reduction

    """

    svd: TruncatedSVD
    """Dimensionality reduction using truncated SVD"""
    right_vector: np.ndarray
    """Temporal structure (right singular vectors). `Array[float, [F, C]]`"""
    left_vector: np.ndarray
    """Spatial structure (left singular vectors). `Array[float, [W * H, C]]`"""

    @property
    def singular_value(self) -> np.ndarray:
        """The singular values corresponding to each of the selected components,
        represent the strength/energy of each component. `Array[float, C]`"""
        return self.svd.singular_values_


def compute_singular_vector(sequences: np.ndarray,
                            n_components: int = 128,
                            **kwargs) -> SequenceSingularVector:
    """
    Performs truncated singular value decomposition (SVD) on a sequence of image frames
    to extract dominant spatial and temporal components.

    :param sequences: A numpy array representing a collection of image frames in a sequence.
                      It has a shape of (n_frames, width, height) where 'n_frames' is the number
                      of frames, and 'width' and 'height' are the dimensions of each frame.
    :param n_components: An integer representing the number of components for Truncated SVD.
                         The default value is 128.
    :param kwargs: Keyword arguments passed to ``TruncatedSVD()``.
    :return: A SequenceSingularVector object containing the singular values, the transformed
             components, and the left singular vectors.
    """
    n_frames, width, height = sequences.shape
    seq = sequences.reshape(n_frames, height * width).T
    # subtracts the mean over time for each pixel
    seq = seq - np.mean(seq, axis=1, keepdims=True)

    svd = TruncatedSVD(n_components=n_components, **kwargs)
    Vsv = svd.fit_transform(seq.T)
    U = seq @ (Vsv / (Vsv ** 2).sum(axis=0) ** 0.5)
    U /= (U ** 2).sum(axis=0) ** 0.5

    return SequenceSingularVector(svd, Vsv, U)
