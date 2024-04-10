from typing import cast

import attrs
import numpy as np

__all__ = [
    'RigEvent',
    'CamEvent'
]


@attrs.define
class RigEvent:
    """container for riglog event type/value"""

    name: str
    """event name"""
    data: np.ndarray
    """2d array (T, V)"""

    def __attrs_post_init__(self):
        if len(self.time) != len(self.value):
            raise RuntimeError('')

    @property
    def time(self) -> np.ndarray:
        return self.data[:, 0]

    @property
    def value(self) -> np.ndarray:
        return self.data[:, 1]

    @property
    def start_time(self) -> float:
        return cast(float, self.time[0])

    @property
    def end_time(self) -> float:
        return cast(float, self.time[-1])

    @property
    def value_index(self) -> np.ndarray:
        """i.e., lap_value, which start accumulating from 1, then turn to 0-base"""
        if np.all(np.diff(self.value)) == 1:
            return (self.value - 1).astype(int)

        raise RuntimeError('incorrect operation')

    def with_pseudo_value(self, value: float) -> 'RigEvent':
        """pseudo value. i.e., for scatter"""
        t = self.time
        v = np.full_like(t, value, dtype=np.double)
        return attrs.evolve(self, data=np.column_stack([t, v]))


@attrs.frozen
class CamEvent(RigEvent):
    """container for riglog camera event type/value"""

    def __len__(self):
        """n_pulses"""
        return len(self.value)

    @property
    def n_pulses(self) -> int:
        """number of camera pulse"""
        return len(self)

    @property
    def fps(self) -> float:
        """approximate median fps in hz"""
        ret = np.median(1 / np.diff(self.time))
        return cast(float, ret)

