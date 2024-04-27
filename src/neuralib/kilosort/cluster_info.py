from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from polars.type_aliases import JoinStrategy

__all__ = ['read_csv']


def read_csv(path: Union[str, Path], **kwargs) -> pl.DataFrame:
    ret = pl.read_csv(path, separator='\t', **kwargs)
    if 'cluster_id' not in ret.columns:
        raise RuntimeError('not a Cluster DataFrame')
    return ret


@pl.api.register_dataframe_namespace("cluster_info")
class ClusterInfoPolarExtension:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    @property
    def cluster_id(self) -> np.ndarray:
        return self._df.get_column('cluster_id').to_numpy()

    def sort_by_cluster_id(self) -> pl.DataFrame:
        return self._df.sort('cluster_id', nulls_last=True)

    def filter_clusters(self, cluster: Union[int, list[int], np.ndarray]) -> pl.DataFrame:
        if isinstance(cluster, int) or np.isscalar(cluster):
            return self._df.filter(pl.col('cluster_id') == cluster)
        elif isinstance(cluster, np.ndarray):
            cluster = list(cluster)

        return self._df.filter(pl.col('cluster_id').is_in(cluster))

    def add_column(self, name: str, value: np.ndarray, cluster: np.ndarray = None) -> pl.DataFrame:
        if name in self._df.columns:
            raise RuntimeError(f'column {name} has existed')

        if cluster is None:
            return self._df.with_columns(pl.lit(value).alias(name))

        dat = pl.DataFrame({'cluster_id': cluster, name: value})
        return self._df.join(dat, on='cluster_id', how='left')

    def join(self, df: pl.DataFrame, how: JoinStrategy = 'left', **kwargs) -> pl.DataFrame:
        if 'cluster_id' not in df.columns:
            raise RuntimeError('not a ClusterInfo DataFrame')

        return self._df.join(df, on='cluster_id', how=how, **kwargs)

    def filter_labels(self, labels: list[str], label_column: str = 'group') -> pl.DataFrame:
        if len(labels) == 1:
            return self._df.filter(pl.col(label_column).cast(pl.Utf8) == labels[0])
        else:
            return self._df.filter(pl.col(label_column).cast(pl.Utf8).is_in(labels))

    def save(self, path: Union[str, Path]):
        self._df.write_csv(Path(path), separator='\t')
