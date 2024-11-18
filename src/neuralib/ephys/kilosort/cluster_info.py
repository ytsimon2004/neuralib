from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, TYPE_CHECKING

import numpy as np
import polars as pl
from typing_extensions import Self, overload

from neuralib.util.util_polars import DataFrameWrapper, helper_with_index_column

if TYPE_CHECKING:
    from polars import _typing as pty
    from .result import KilosortResult
    from neuralib.ephys.glx import EphysRecording

__all__ = ['ClusterInfo']


class ClusterInfo(DataFrameWrapper):
    """
    Cluster info data frame. Any polars with the column "cluster_id" could be considered `ClusterInfo`.

    It sometimes contains `ChannelInfo`'s columns: `[cluster_id, channel, shank, pos_x, pos_y]`, which channel
    is the most significant channel of corresponding clusters. However, depending on the methods,
    what the most significant channel to a cluster is may be changed. For example, waveform-template-based and
    mean-waveform-based methods may produce slightly different results. Note that, when we talk about the
    channel of a cluster, we only focus on non-noise clusters.
    """

    def __init__(self, df: pl.DataFrame | DataFrameWrapper):
        """
        Wrap a dataframe as a `ClusterInfo`.

        :raise RuntimeError: *df* does not contain "cluster_id" column.
        """
        if isinstance(df, DataFrameWrapper):
            df = df.dataframe()

        if 'cluster_id' not in df.columns:
            raise RuntimeError('not a cluster info dataframe. miss "cluster_id" column.')

        self._df = df
        self.use_label_column = 'label'

    @classmethod
    @overload
    def read_csv(cls, path: str | Path, *,
                 # change default values
                 separator: str = ",",
                 comment_prefix: str | None = '#',
                 quote_char: str | None = '"',
                 skip_rows: int = 0,
                 # common kwargs
                 schema_overrides: Mapping[str, pty.PolarsDataType] | Sequence[pty.PolarsDataType] = None,
                 null_values: str | Sequence[str] | dict[str, str] | None = None,
                 # other kwargs
                 **kwargs) -> ClusterInfo:
        pass

    @classmethod
    def read_csv(cls, path: str | Path, *,
                 separator: str = "\t",
                 comment_prefix: str | None = '#',
                 quote_char: str | None = '"',
                 skip_rows: int = 0,
                 **kwargs) -> ClusterInfo:
        """
        Read cluster csv dataframe.

        It used to load Kilosort/Phy csv/tsv data, which has the following form.
        
        ```
        $ head cluster_DATA.csv
        cluster_id  DATA
        0   0.0
        1   1.0
        ```

        :param path:
        :param separator:
        :param comment_prefix:
        :param quote_char:
        :param skip_rows:
        :param kwargs: polars.read_csv(kwargs)
        :return:
        """
        return ClusterInfo(pl.read_csv(
            path,
            separator=separator,
            comment_prefix=comment_prefix,
            quote_char=quote_char,
            skip_row=skip_rows,
            **kwargs
        ))

    def save(self, path: str | Path | IO | None, *,
             separator: str = "\t",
             quote_char: str = '"', ):
        self._df.write_csv(path, separator=separator, quote_char=quote_char)

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

        :param cluster: cluster ID, ID list, ID array, or a ClusterInfo.
        :param maintain_order: keep the ordering of *cluster* in the returned dataframe.
        :param strict: all *cluster* should present in the returned dataframe. Otherwise, an error will be raised.
        :return:
        :raise RuntimeError: strict mode fail.
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

    def join(self, other: pl.DataFrame | DataFrameWrapper, on='cluster_id', **kwargs) -> Self:
        return super().join(other, on, **kwargs)

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
        Filter clusters with given labels.

        :param labels:
        :param column: the name of the label column. default use `self.use_label_column`,
        :return: self that only contains clusters with label in *labels*.
        :raise ColumnNotFoundError: *column*
        """
        if column is None:
            column = self.use_label_column

        if isinstance(labels, str):
            return self.filter(pl.col(column) == labels)
        else:
            return self.filter(pl.col(column).is_in(labels))


def fix_cluster_info(info: ClusterInfo,
                     ephys: EphysRecording | None = None,
                     ks_result: KilosortResult | None = None) -> ClusterInfo:
    """

    This method will try to fix following things:

    * change column name, such as 'ch' to 'channel', 'sh' to 'shank'
    * map channel from channel index to channel number (*ks_result* is not None)
    * replace with actual shank number (*ephys* is not None)

    TODO check kilosort/phy still contain above issues.

    :param info:
    :param ephys:
    :param ks_result:
    :return:
    """
    ret = info

    # rename columns
    rename = {}
    if 'ch' in ret.columns:
        rename['ch'] = 'channel'
    if 'sh' in ret.columns:
        rename['sh'] = 'shank'
    if 'depth' in ret.columns:
        rename['depth'] = 'pos_y'

    if len(rename):
        ret = ret.rename(rename)

    if ks_result is not None:
        chmap = ks_result.channel_map
        chmap = pl.DataFrame(dict(channel=np.arange(chmap), channel_value=chmap))
        ret = (
            ret
            .join(chmap, on='channel', how='left')
            .drop('channel')
            .rename({'channel_value': 'channel'})
        )

    if ephys:
        ret = (
            ret
            .join(ephys.channel_info(), on='channel', how='left')
            .drop('shank', 'pos_y')
            .rename({
                'shank_right': 'shank',
                'pos_y_right': 'pos_y',
            })
        )

    return ret
