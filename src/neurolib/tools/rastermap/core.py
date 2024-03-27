from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import attrs
import numpy as np

from rscvp.util.util_verbose import fprint

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
    """number of clusters to compute"""
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
    filename: str
    save_path: str
    isort: np.ndarray
    """(N',)"""
    embedding: np.ndarray
    """(N',1)"""
    ops: RasterOptions

    user_clusters: list[UserCluster] = attrs.field(default=attrs.Factory(list))

    # customized
    super_neurons: np.ndarray | None = attrs.field(default=None)

    @classmethod
    def load(cls, path: Path) -> 'RasterMapResult':
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
