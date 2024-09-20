from __future__ import annotations

import numpy as np
import polars as pl
from typing_extensions import Self

from neuralib.util.util_polars import DataFrameWrapper

__all__ = ['ChannelInfo']


class ChannelInfo(DataFrameWrapper):
    """
    Channel info data frame. Any polars with a columns "channel" could be considered ChannelInfo.

    It usually contains columns: [channel, shank, pos_x, pos_y]
    """

    def __init__(self, df: pl.DataFrame):
        if 'channel' not in df.columns:
            raise RuntimeError('not a cluster info. miss "channel" column.')

        self._df = df

    def dataframe(self, dataframe: pl.DataFrame = None, may_inplace=True):
        if dataframe is None:
            return self._df
        else:
            ret = ChannelInfo(dataframe)
            return ret

    """property"""

    @property
    def channel(self) -> np.ndarray:
        return self['channel'].to_numpy()

    @property
    def n_shanks(self) -> int:
        return self['shank'].n_unique()

    @property
    def shank_set(self) -> np.ndarray:
        return self['shank'].unique().to_numpy()

    @property
    def shank(self) -> np.ndarray:
        return self['shank'].to_numpy()

    @property
    def pos_x(self) -> np.ndarray:
        return self['pos_x'].to_numpy()

    @property
    def pos_y(self) -> np.ndarray:
        return self['pos_y'].to_numpy()

    """channels"""

    def with_channels(self, channel: list[int] | np.ndarray | pl.Series) -> Self:
        index = pl.DataFrame(
            dict(channel=channel),
            schema_overrides=dict(channel=self._df.schema['channel'])
        ).with_row_index('_index')
        ret = self.lazy().join(index, on=['channel'], how='left')
        ret = ret.filter(pl.col('_index').is_not_null())
        return ret.sort('_index').drop('_index').collect()

    def filter_channels(self, channel: int | list[int] | np.ndarray) -> Self:
        channel = np.atleast_1d(channel)
        return self.filter(pl.col('channel').is_in(channel))

    def drop_channels(self, channel: int | list[int] | np.ndarray) -> Self:
        channel = np.atleast_1d(channel)
        return self.filter(pl.col('channel').is_in(channel).not_())

    def filter_shanks(self, shank: int | list[int] | np.ndarray) -> Self:
        shank = np.atleast_1d(shank)
        return self.filter(pl.col('shank').is_in(shank))

    def sort_channels(self, descending: bool = False) -> Self:
        return self.sort('channel', descending=descending)

    def sort_channel_by_depth(self, descending: bool = False) -> Self:
        return self.sort('pos_y', descending=descending)

    """position"""

    def mm(self) -> Self:
        if len(self) == 0:
            return self

        if self['pos_y'].max() < 10:
            return self

        columns = dict(pos_y=pl.col('pos_y') / 1000)
        if 'pos_x' in self._df.columns:
            columns['pos_x'] = pl.col('pos_x') / 1000

        return self.with_columns(**columns)

    def um(self) -> Self:
        if len(self) == 0:
            return self

        if self['pos_y'].max() > 100:
            return self

        columns = dict(pos_y=pl.col('pos_y') * 1000)
        if 'pos_x' in self._df.columns:
            columns['pos_x'] = pl.col('pos_x') * 1000

        return self.with_columns(**columns)

    """other"""

    def partition_channel_by_shank(self) -> dict[int, Self]:
        return {
            shank: data
            for (shank,), data in self.partition_by('shank', as_dict=True)
        }
