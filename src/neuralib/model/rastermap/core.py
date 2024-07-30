from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import attrs
import numpy as np
from typing_extensions import Self

from neuralib.util.util_verbose import fprint

__all__ = ['UserCluster',
           'RasterOptions',
           'RasterMapResult']


class UserCluster(TypedDict, total=False):
    """GUI selected clusters"""

    ids: np.ndarray
    """neuronal ids"""
    slice: slice
    """binned neurons range"""
    binsize: int
    """neuron bins"""
    color: np.ndarray


class RasterOptions(TypedDict, total=False):
    """Run Rastermap model options"""

    n_clusters: int
    """number of clusters to compute. TODO check how affect the result"""
    n_PCs: int
    """number of PCs to use"""
    time_lag_window: float
    """use future timepoints to compute correlation"""
    locality: float
    """locality in sorting to find sequences"""
    n_splits: int
    """TODO"""
    time_bin: int
    """TODO"""
    grid_upsample: int
    """default value, 10 is good for large recordings"""

    mean_time: bool
    verbose: bool
    verbose_sorting: bool
    end_time: int
    start_time: int


@attrs.define
class RasterMapResult:
    """Container for storing the rastermap result,
    For both GUI load and customized plotting purpose

    `Dimension parameters`:

        N = number of neurons

        T = number of image pulse

        C = number of clusters = N / binsize

    """

    filename: str
    """neural activity filename (N, T)"""
    save_path: str
    """filename for the rastermap result save"""
    isort: np.ndarray
    """(N,)"""
    embedding: np.ndarray
    """(N,1)"""
    ops: RasterOptions
    """`RasterOptions`"""
    user_clusters: list[UserCluster] = attrs.field(default=attrs.Factory(list))
    """list of clusters `UserCluster`"""

    super_neurons: np.ndarray | None = attrs.field(default=None)
    """super neuron activity (C, T)"""

    @classmethod
    def load(cls, path: Path) -> Self:
        """
        Load the results from rastermap output

        :param path: file path of the rastermap output
        :return: :class:`RasterMapResult`
        """
        dat = np.load(path, allow_pickle=True).item()
        fprint(f'LOAD ->{path}', vtype='io')

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
        fprint(f'SAVE ->{path}', vtype='io')

    @property
    def n_super(self) -> int:
        """n_clusters"""
        return len(self.super_neurons)
