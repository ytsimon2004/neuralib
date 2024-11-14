import abc
from pathlib import Path
from typing import Optional

import numpy as np

from neuralib.ephys.glx.channel_info import ChannelInfo

__all__ = ['EphysRecording']


class EphysRecording(metaclass=abc.ABCMeta):
    """
    An ephys recording data that contains `C` channels and `T` samples.
    """

    @property
    @abc.abstractmethod
    def data_path(self) -> Optional[Path]:
        """
        The path to the data file.

        :return: The path to the data file or None if not set.
        """
        pass

    @property
    def total_channels(self) -> int:
        """
        Total number of channels in the recording.

        :return: Number of channels in the recording.
        """
        return len(self.channel_list)

    @property
    def channel_list(self) -> np.ndarray:
        """
        List of channel indices.

        :return: Array of channel indices.
        """
        return np.arange(self.total_channels)

    @property
    @abc.abstractmethod
    def total_samples(self) -> int:
        """
        Total number of samples in the recording.

        :return: Number of samples in the recording.
        """
        pass

    @property
    @abc.abstractmethod
    def sample_rate(self) -> float:
        """
        Sampling rate of the recording.

        :return: The sampling rate in Hz.
        """
        pass

    @property
    def total_duration(self) -> float:
        """
        Total duration of the recording in seconds.

        :return: The duration of the recording in seconds.
        """
        return self.total_samples / self.sample_rate

    @property
    def time_start(self) -> float:
        """
        Start time of the recording.

        :return: The start time of the recording, defaults to 0.
        """
        return 0

    @property
    def t(self) -> np.ndarray:
        """
        Array of time points corresponding to each sample.

        :return: Array of time points.
        """
        return np.linspace(0, self.total_duration, self.total_samples) + self.time_start

    @abc.abstractmethod
    def __getitem__(self, item) -> np.ndarray:
        pass

    @property
    def meta(self) -> dict[str, str]:
        """
        Metadata associated with the recording.

        :return: A dictionary containing metadata key-value pairs.
        """
        return {}

    def channel_info(self) -> ChannelInfo:
        """
        Get information about the channels.

        :raises NotImplementedError: This class not support this method.
        :return: An object containing channel information.
        """
        raise NotImplementedError()
