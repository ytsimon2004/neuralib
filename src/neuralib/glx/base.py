import abc
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = ['EphysRecording']


class EphysRecording(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def data_path(self) -> Optional[Path]:
        pass

    @property
    def total_channels(self) -> int:
        return len(self.channel_list)

    @property
    def channel_list(self) -> np.ndarray:
        return np.arange(self.total_channels)

    @property
    @abc.abstractmethod
    def total_samples(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def sample_rate(self) -> float:
        pass

    @property
    def total_duration(self) -> float:
        return self.total_samples / self.sample_rate

    @property
    def time_start(self) -> float:
        return 0

    @property
    def t(self) -> np.ndarray:
        return np.linspace(0, self.total_duration, self.total_samples) + self.time_start

    @abc.abstractmethod
    def __getitem__(self, item) -> np.ndarray:
        pass

    @property
    def meta(self) -> dict[str, str]:
        return {}

