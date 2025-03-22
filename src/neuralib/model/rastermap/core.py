from pathlib import Path
from typing import TypedDict

import attrs
import numpy as np
from neuralib.typing import PathLike
from neuralib.util.verbose import print_save, print_load
from typing_extensions import Self

__all__ = ['UserCluster',
           'RasterOptions',
           'RasterMapResult']


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
    end_time: int


@attrs.define
class RasterMapResult:
    """Container for storing the rastermap result,
    For both GUI load and customized plotting purpose

    `Dimension parameters`:

        N = Number of neurons/pixel

        T = Number of image pulse

        C = Number of clusters = N / binsize

    """

    filename: str
    """
    Filename of the neural activity data
    (i.e., *.tif or *.avi for wfield activity; .npy `Array[float, [N, T]]` file for cellular)
    """

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
            filename=dat['filename'],
            save_path=dat['save_path'],
            isort=dat['isort'],
            embedding=dat['embedding'],
            ops=dat['ops'],
            user_clusters=dat['user_clusters'],
            super_neurons=dat['super_neurons']
        )

    def save(self, path: Path) -> None:
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

    @property
    def n_super(self) -> int:
        """number of clusters"""
        return len(self.super_neurons)
