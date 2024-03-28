from __future__ import annotations

from typing import NamedTuple, Any

import numpy as np
from typing_extensions import TypeAlias

__all__ = ['Session',
           'SessionInfo']

Session: TypeAlias = str


class SessionInfo(NamedTuple):
    name: Session
    """name of this session"""

    time: tuple[float, float]
    """time range of this session"""

    def time_mask_of(self, t: np.ndarray) -> np.ndarray:
        """
        create a mask for time array *t*.

        :param t: 1d time array
        :return: mask for this session
        """
        return np.logical_and(self.time[0] < t, t < self.time[1])

    def in_range(self, time: np.ndarray,
                 value: np.ndarray = None,
                 error=True) -> tuple[Any, Any]:
        """
        Get the range (the first and last value) of value array in this session.

        :param time: 1d time array
        :param value: 1d value array. Shape should as same as *time*
        :param error: raise an error when empty.
        :return: tuple of first and last `value` or `time`.
        """
        x = self.time_mask_of(time)
        if value is not None:
            t = value[x]
        else:
            t = time[x]

        if len(t) == 0:
            if error:
                raise ValueError('empty in extracting value or time from time mask')
            return np.nan, np.nan

        return t[0], t[-1]

    def in_slice(self, time: np.ndarray,
                 value: np.ndarray,
                 error=True) -> slice:
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError()

        v = value[self.time_mask_of(time)]

        if len(v) == 0:
            if error:
                raise ValueError('empty in extracting value or time from time mask')
            return slice(0, 0)

        return slice(int(v[0]), int(v[-1]))
