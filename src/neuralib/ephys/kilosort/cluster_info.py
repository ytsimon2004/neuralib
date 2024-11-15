from __future__ import annotations

from pathlib import Path
from typing import IO

import numpy as np
import polars as pl
from typing_extensions import Self

from neuralib.util.util_polars import DataFrameWrapper, helper_with_index_column

__all__ = ['ClusterInfo']


class ClusterInfo(DataFrameWrapper):
    """
    Cluster info data frame. Any polars with a columns "cluster_id" could be considered ClusterInfo.
    """

    def __init__(self, df: pl.DataFrame | DataFrameWrapper):
        """
        Wrap a dataframe as a ClusterInfo.

        :raise RuntimeError: *df* does not contain "cluster_id" column.
        """
        if isinstance(df, DataFrameWrapper):
            df = df.dataframe()

        if 'cluster_id' not in df.columns:
            raise RuntimeError('not a cluster info. miss "cluster_id" column.')

        self._df = df
        self.use_label_column = 'label'

    @classmethod
    def read_csv(cls, path: str | Path, **kwargs) -> ClusterInfo:
        ret = pl.read_csv(path, separator='\t', **kwargs)
        return ClusterInfo(ret)

    def save(self, path: str | Path | IO | None):
        self._df.write_csv(path, separator='\t')

    def dataframe(self, dataframe: pl.DataFrame = None, may_inplace=True):
        if dataframe is None:
            return self._df
        else:
            ret = ClusterInfo(dataframe)
            ret.use_label_column = self.use_label_column
            return ret

    """property"""

    @property
    def cluster_id(self) -> np.ndarray:
        """
        :return: cluster id `Array[int, C]`
        """
        return self._df.get_column('cluster_id').to_numpy()

    @property
    def cluster_shank(self) -> np.ndarray:
        """
        shank array of clusters.

        :return: shank array `Array[shank:int, C]`
        :raise ColumnNotFoundError: shank
        """
        return self['shank'].to_numpy()

    @property
    def cluster_channel(self) -> np.ndarray:
        """
        channel (the significant channel) array of clusters

        :return: channel array `Array[channel:int, C]`
        :raise ColumnNotFoundError: channel
        """
        return self['channel'].to_numpy()

    @property
    def cluster_pos_x(self) -> np.ndarray:
        """
        :return: channel x position `Array[float, C]`.
        :raise ColumnNotFoundError: pos_x
        """
        return self['pos_x'].to_numpy()

    @property
    def cluster_pos_y(self) -> np.ndarray:
        """
        :return: channel y position `Array[float, C]`
        :raise ColumnNotFoundError: pos_y
        """
        return self['pos_y'].to_numpy()

    """clusters"""

    def with_clusters(self, cluster: int | list[int] | np.ndarray | ClusterInfo, *,
                      maintain_order: bool = False,
                      strict: bool = False) -> Self:
        """
        select particular clusters and keep the ordering of *cluster*.

        :param cluster:
        :param maintain_order: keep the ordering of *cluster* in the returned dataframe.
        :param strict: all *cluster* should present in the returned dataframe. Otherwise, an error will be raised.
        :return:
        """
        return helper_with_index_column(self, 'cluster_id', cluster, maintain_order, strict)

    def sort_cluster_by_id(self) -> Self:
        return self.sort('cluster_id', nulls_last=True)

    def sort_cluster_by_depth(self, descending: bool = False) -> Self:
        """
        sort clusters by their depth.

        :return:
        :raise ColumnNotFoundError: depth
        """
        return self.sort('depth', descending=descending, nulls_last=True)

    def append_cluster_data(self, cluster_id: int, **kwargs) -> Self:
        """
        append a new cluster data.

        :param cluster_id:
        :param kwargs:
        :return:
        """
        if np.any(self.cluster_id == cluster_id):
            raise RuntimeError(f'cluster {cluster_id} has been used')

        other = pl.DataFrame({'cluster_id': cluster_id, **kwargs}, schema=self._df.schema)
        return self.dataframe(pl.concat([self._df, other], how='diagonal'))

    def join(self, other: pl.DataFrame | DataFrameWrapper, on='cluster_id', how="inner", **kwargs) -> Self:
        return super().join(other, on, how, **kwargs)

    """label"""

    def n_cluster_label(self, label: str, column: str = None) -> int:
        """
        number of cluster labeled as *label*.

        :param label:
        :param column: label column. default use `self.use_label_column`,
        :return: count
        :raise ColumnNotFoundError: *column*
        """
        if column is None:
            column = self.use_label_column

        return np.count_nonzero((self[column] == label).to_numpy())

    def with_cluster_label(self, labels: str | list[str], column: str = None) -> Self:
        """

        :param labels:
        :param column: label column. default use `self.use_label_column`,
        :return:
        :raise ColumnNotFoundError: *column*
        """
        if column is None:
            column = self.use_label_column

        if isinstance(labels, str):
            return self.filter(pl.col(column) == labels)
        else:
            return self.filter(pl.col(column).is_in(labels))
