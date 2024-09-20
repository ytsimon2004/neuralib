from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Union

__all__ = ['KilosortParameter']


class KilosortParameter(NamedTuple):
    """data for kilosort params.py file"""

    para_file: Path

    data_path: Path
    """location of raw data file"""

    channel_number: int
    """total number of rows in the data file (not just those that have your neural data on them. 
    This is for loading the file)"""

    data_type: str
    """data type to read, e.g. 'int16'"""

    channel_offset: int
    """number of bytes at the beginning of the file to skip"""

    sample_rate: float
    """ in Hz"""

    hp_filtered: bool
    """True/False, whether the data have already been filtered"""

    @classmethod
    def of(cls, path: Union[str, Path]) -> KilosortParameter:
        var = {}
        exec(Path(path).read_text(), {}, var)

        data_file = var["dat_path"]
        data_file = path.with_name(data_file) if '/' in data_file else Path(data_file)
        channel_number = int(var["n_channels_dat"])
        data_type = var["dtype"]
        channel_offset = var["offset"]
        sample_rate = float(var["sample_rate"])
        hp_filtered = bool(var["hp_filtered"])
        return KilosortParameter(path, data_file, channel_number, data_type, channel_offset, sample_rate, hp_filtered)

    @classmethod
    def get_total_channels(cls, path: Union[str, Path]) -> int:
        var = {}
        exec(Path(path).read_text(), {}, var)
        return int(var["n_channels_dat"])

    @classmethod
    def get_data_path(cls, path: Union[str, Path]) -> Path:
        var = {}
        exec(Path(path).read_text(), {}, var)
        data_file = var["dat_path"]
        return path.with_name(data_file) if '/' in data_file else Path(data_file)

    def with_data_path(self, data_path: Path) -> KilosortParameter:
        return self._replace(data_path=data_path)
