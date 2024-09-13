from __future__ import annotations

from pathlib import Path
from typing import IO

import numpy as np
import polars as pl
from typing_extensions import Self

from neuralib.util.util_polars import DataFrameWrapper

__all__ = ['ClusterInfo']


class ClusterInfo(DataFrameWrapper):
    """
    Cluster info data frame. Any polars with a columns "cluster_id" could be considered ClusterInfo.
    """

    def __init__(self, df: pl.DataFrame):
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
        return self._df.get_column('cluster_id').to_numpy()

    @property
    def cluster_shank(self) -> np.ndarray:
        """
        shank array of clusters.

        :return:
        :raise ColumnNotFoundError: sh
        """
        return self['sh'].to_numpy()

    @property
    def cluster_channel(self) -> np.ndarray:
        """

        :return:
        :raise ColumnNotFoundError: ch
        """
        return self['ch'].to_numpy()

    @property
    def cluster_pos_y(self) -> np.ndarray:
        """
        :return:
        :raise ColumnNotFoundError: depth
        """
        return self['depth'].to_numpy()

    """clusters"""

    def with_clusters(self, cluster: list[int] | np.ndarray | ClusterInfo) -> Self:
        """
        select particular clusters and keep the ordering of *cluster*.

        This method does not ensure every cluster in this are present in *cluster*.

        :param cluster:
        :return:
        """
        if isinstance(cluster, ClusterInfo):
            cluster_id = cluster.cluster_id
        else:
            cluster_id = np.asarray(cluster)

        index = pl.DataFrame(
            dict(cluster_id=cluster_id),
            schema_overrides=dict(cluster_id=self._df.schema['cluster_id'])
        ).with_row_index('_index')
        ret = self.lazy().join(index, on='cluster_id', how='left')
        ret = ret.filter(pl.col('_index').is_not_null())
        return ret.sort('_index').drop('_index').collect()

    def filter_clusters(self, cluster: int | list[int] | np.ndarray | ClusterInfo) -> Self:
        """
        select particular clusters.

        :param cluster:
        :return:
        """
        if isinstance(cluster, (int, np.integer)):
            return self.filter(pl.col('cluster_id') == cluster)

        if isinstance(cluster, ClusterInfo):
            cluster = cluster.cluster_id

        return self._df.filter(pl.col('cluster_id').is_in(cluster))

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

    def join(self, other: pl.DataFrame, on='cluster_id', how="inner", *,
             left_on=None,
             right_on=None,
             suffix: str = "_right",
             validate="m:m",
             join_nulls: bool = False,
             coalesce: bool | None = None) -> Self:
        return super().join(other, on, how, left_on=left_on, right_on=right_on, suffix=suffix,
                            validate=validate, join_nulls=join_nulls, coalesce=coalesce)

    """label"""

    def n_cluster_label(self, label: str, column: str = None) -> int:
        """
        number of cluster labeled as *label*.

        :param label:
        :param column:
        :return: count
        :raise ColumnNotFoundError: *column*
        """
        if column is None:
            column = self.use_label_column

        return np.count_nonzero((self[column] == label).to_numpy())

    def filter_cluster_label(self, labels: str | list[str], column: str = None) -> Self:
        """

        :param labels:
        :param column:
        :return:
        :raise ColumnNotFoundError: *column*
        """
        if column is None:
            column = self.use_label_column

        if isinstance(labels, str):
            return self.filter(pl.col(column) == labels)
        else:
            return self.filter(pl.col(column).is_in(labels))
