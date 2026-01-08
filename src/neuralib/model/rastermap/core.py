from __future__ import annotations

from os import PathLike

import attrs
import numpy as np
from typing import Self
from typing import TypedDict, Any

from neuralib.util.verbose import print_load, print_save

__all__ = ['read_rastermap',
           'save_rastermap',
           'RasterMapResult',
           'UserCluster',
           'RasterOptions']


def read_rastermap(file: PathLike) -> RasterMapResult:
    """load rastermap result
    :param file: output from rastermap. *_embedding.npy
    :return:
    """
    return RasterMapResult.load(file)


def save_rastermap(result: RasterMapResult, path: PathLike):
    """save result for GUI relaunch

    :param result: :class:`~RasterMapResult`
    :param path: output from rastermap. *_embedding.npy
    """
    result.save(path)


@attrs.define
class RasterMapResult:
    """Container for storing the rastermap result,
    For both GUI load and customized plotting purpose

    `Dimension parameters`:

        N = Number of neurons/pixel

        T = Number of samples

        C = Number of clusters = N / binsize

    """

    filename: str
    """Filename of the neural activity data
    (i.e., *.tif or *.avi for wfield activity; .npy `Array[float, [N, T]]` file for cellular)"""

    save_path: str
    """filename for the rastermap result save"""

    isort: np.ndarray
    """`Array[int, N]`"""

    embedding: np.ndarray
    """`Array[float, [N, 1]]`"""

    ops: RasterOptions
    """``RasterOptions``"""

    user_clusters: list[UserCluster] = attrs.field(default=attrs.Factory(list))
    """list of clusters ``UserCluster``"""

    super_neurons: np.ndarray | None = attrs.field(default=None)
    """super neuron activity. `Array[float, [C, T]]`"""

    @classmethod
    def load(cls, path: PathLike) -> Self:
        """
        Load the results from rastermap output

        :param path: file path of the rastermap output
        :return: :class:`RasterMapResult`
        """
        dat = np.load(path, allow_pickle=True).item()
        print_load(path)

        return RasterMapResult(
            filename=dat.get('filename', None),
            save_path=dat.get('save_path', None),
            isort=dat.get('isort', None),
            embedding=dat.get('embedding', None),
            ops=dat.get('ops', None),
            user_clusters=dat.get('user_clusters', None),
            super_neurons=dat.get('super_neurons', None)
        )

    def save(self, path: PathLike) -> None:
        """For GUI loading & cache computing for plotting in different time domains"""
        proc = {
            'filename': self.filename,
            'save_path': self.save_path,
            'isort': self.isort,
            'embedding': self.embedding,
            'ops': self.ops,
            'user_clusters': self.user_clusters,
            'super_neurons': self.super_neurons,
        }

        np.save(path, proc, allow_pickle=True)
        print_save(path)

    def asdict(self) -> dict[str, Any]:
        """result as dict"""
        return attrs.asdict(self)

    @property
    def n_clusters(self) -> int:
        """number of clusters (super neurons)"""
        try:
            return self.super_neurons.shape[0]
        except AttributeError:
            raise RuntimeError('data incomplete')

    @property
    def n_samples(self) -> int:
        """number of data samples (T)"""
        try:
            return self.super_neurons.shape[1]
        except AttributeError:
            raise RuntimeError('data incomplete')


class UserCluster(TypedDict, total=False):
    """GUI selected clusters"""
    ids: np.ndarray
    """Neuronal ids. `Array[int, N]`"""
    slice: slice
    """Binned neurons range"""
    binsize: int
    """Neuron bins"""
    color: np.ndarray
    """Colors. `Array[float, 4]`"""


class RasterOptions(TypedDict, total=False):
    """Run Rastermap model options. Refer to the ``rastermap.rastermap.setting_info()``"""

    n_clusters: int
    """Number of clusters created from data before upsampling and creating embedding 
    (any number above 150 will be very slow due to NP-hard sorting problem)"""

    n_PCs: int
    """Number of PCs to use during optimization"""

    time_lag_window: float
    """Number of time points into the future to compute cross-correlation, useful for sequence finding"""

    locality: float
    """How local should the algorithm be -- set to 1.0 for highly local + sequence finding"""

    n_splits: int
    """Recluster and sort n_splits times (increases local neighborhood preservation)"""

    time_bin: int
    """Binning of data in time before PCA is computed"""

    grid_upsample: int
    """How much to upsample clusters"""

    mean_time: bool
    """Whether to project out the mean over data samples at each timepoint, usually good to keep on to find structure"""

    verbose: bool
    """Whether to output progress during optimization"""

    verbose_sorting: bool
    """Output progress in travelling salesman"""

    start_time: int
    """Optional start time"""

    end_time: int
    """Optional end time"""
