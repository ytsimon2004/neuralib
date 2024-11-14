from __future__ import annotations

import numpy as np
import polars as pl
from typing_extensions import Self

from neuralib.util.util_polars import DataFrameWrapper, helper_with_index_column

__all__ = ['ChannelInfo']


class ChannelInfo(DataFrameWrapper):
    """
    Channel info data frame. Any polars with a columns "channel" could be considered a `ChannelInfo`.

    It usually contains columns: [channel, shank, pos_x, pos_y]
    """

    def __init__(self, df: pl.DataFrame):
        """
        Wrap a dataframe as a ChannelInfo.

        :raise RuntimeError: *df* does not contain "channel" column.
        """
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
        """
        :return: channel `Array[int, C]`.
        """
        return self['channel'].to_numpy()

    @property
    def n_shanks(self) -> int:
        """

        :return:
        :raise ColumnNotFoundError: shank
        """
        return self['shank'].n_unique()

    @property
    def shank_set(self) -> np.ndarray:
        """

        :return: unique shank `Array[shank:int, S]`
        :raise ColumnNotFoundError: shank
        """
        return self['shank'].unique().to_numpy()

    @property
    def shank(self) -> np.ndarray:
        """

        :return: shank `Array[shank:int, C]`
        :raise ColumnNotFoundError: shank
        """
        return self['shank'].to_numpy()

    @property
    def pos_x(self) -> np.ndarray:
        """
        :return: channel x position `Array[float, C]`
        :raise ColumnNotFoundError: pos_x
        """
        return self['pos_x'].to_numpy()

    @property
    def pos_y(self) -> np.ndarray:
        """
        :return: channel y position `Array[float, C]`.
        :raise ColumnNotFoundError: pos_y
        """
        return self['pos_y'].to_numpy()

    """channels"""

    def with_channels(self, channel: int | list[int] | np.ndarray | ChannelInfo, *,
                      maintain_order: bool = False,
                      strict: bool = False) -> Self:
        """
        Restrict channels in this dataframe.

        :param channel:
        :param maintain_order: keep the ordering of *channel* in the returned dataframe.
        :param strict: all *channel* should present in the returned dataframe. Otherwise, an error will be raised.
        :return:
        """
        return helper_with_index_column(self, 'channel', channel, maintain_order, strict)

    def drop_channels(self, channel: int | list[int] | np.ndarray) -> Self:
        """ Remove channels from this dataframe."""
        channel = np.atleast_1d(channel)
        return self.filter(pl.col('channel').is_in(channel).not_())

    def with_shanks(self, shank: int | list[int] | np.ndarray) -> Self:
        """Restrict shanks in this dataframe."""
        shank = np.atleast_1d(shank)
        return self.filter(pl.col('shank').is_in(shank))

    def sort_channels(self, descending: bool = False) -> Self:
        return self.sort('channel', descending=descending)

    def sort_channel_by_depth(self, descending: bool = False) -> Self:
        return self.sort('pos_y', descending=descending)

    """position"""

    def mm(self) -> Self:
        """
        transform the channel position to unit mm.
        do nothing when it has already been mm.

        :return:
        """
        if len(self) == 0:
            return self

        if self['pos_y'].max() < 10:
            return self

        columns = dict(pos_y=pl.col('pos_y') / 1000)
        if 'pos_x' in self._df.columns:
            columns['pos_x'] = pl.col('pos_x') / 1000

        return self.with_columns(**columns)

    def um(self) -> Self:
        """
        transform the channel position to unit um.
        do nothing when it has already been um,

        :return:
        """
        if len(self) == 0:
            return self

        if self['pos_y'].max() > 100:
            return self

        columns = dict(pos_y=pl.col('pos_y') * 1000)
        if 'pos_x' in self._df.columns:
            columns['pos_x'] = pl.col('pos_x') * 1000

        return self.with_columns(**columns)

    """other"""

    def join(self, other: pl.DataFrame | DataFrameWrapper, on='channel', how="inner", **kwargs) -> Self:
        return super().join(other, on, how, **kwargs)

    def partition_channel_by_shank(self) -> dict[int, Self]:
        return {
            shank: data
            for (shank,), data in self.partition_by('shank', as_dict=True)
        }
