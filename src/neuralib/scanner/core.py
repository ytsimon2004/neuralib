from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, NewType, Literal

import numpy as np
from tifffile import tifffile
from typing_extensions import TypeAlias

from neuralib.util.util_type import PathLike
from neuralib.util.util_verbose import fprint

__all__ = [
    #
    'ZPROJ_TYPE',
    'ZEISS_CZI_CHANNELS_ORDER',
    'ZEISS_LSM_CHANNELS_ORDER',
    #
    'SceneIdx',
    'DimCode',
    #
    'AbstractConfocalScanner',
    'parse_tif_meta'
]

ZEISS_CZI_CHANNELS_ORDER = ['Greens', 'Blues', 'Reds']
ZEISS_LSM_CHANNELS_ORDER = ['Blues', 'Reds', 'Greens']

ZPROJ_TYPE = Literal['avg', 'max', 'min', 'std', 'median']

SceneIdx: TypeAlias = int  # 0-base scan position
DimCode = NewType('DIMCODE', str)


class AbstractConfocalScanner(metaclass=abc.ABCMeta):
    """ABC for the confocal image data"""

    n_scenes: int
    """positions scan"""

    meta: dict[str, Any]
    """metadata dict"""

    @classmethod
    @abc.abstractmethod
    def load(cls, filepath: PathLike):
        pass

    @property
    @abc.abstractmethod
    def width(self) -> dict[SceneIdx, int]:
        """X"""
        pass

    @property
    @abc.abstractmethod
    def height(self) -> dict[SceneIdx, int]:
        """Y"""
        pass

    @property
    @abc.abstractmethod
    def n_channels(self) -> dict[SceneIdx, int]:
        """number of fluorescence channels. C"""
        pass

    @property
    @abc.abstractmethod
    def n_zstacks(self) -> dict[SceneIdx, int]:
        """number of stacks in z axis. Z"""
        pass

    @abc.abstractmethod
    def get_dim_code(self) -> DimCode:
        """get the `DimCode`


        `Dimension parameters (DimCode)`:

            V - view

            H - phase

            I - illumination

            S - scene

            R - rotation

            T - time

            C - channel

            Z - z plane (height)

            M - mosaic tile, mosaic images only

            Y - image height

            X - image width

            A - samples, BGR/RGB images only
        """
        pass

    def get_pixel2mm_factor(self):
        pass

    def _zproj(self, stacks: np.ndarray,
               zproj_type: ZPROJ_TYPE = 'max') -> np.ndarray:
        if zproj_type == 'avg':
            img = np.mean(stacks, axis=0)
        elif zproj_type == 'median':
            img = np.median(stacks, axis=0)
        elif zproj_type == 'max':
            img = np.max(stacks, axis=0)
        else:
            raise NotImplementedError('')

        return img


# ===== #

def parse_tif_meta(file: Path, **kwargs) -> dict[str, Any]:
    """
    :return: meta dict
    """
    if file.suffix not in ('.tif', '.tiff', '.lsm'):
        raise ValueError(f'{file.suffix} not support')

    meta_collect = {}
    with tifffile.TiffFile(file, **kwargs) as tif:
        for attr in dir(tif):
            if 'meta' in attr:
                meta = getattr(tif, attr)
                if meta is not None:
                    if isinstance(meta, dict):  # i.e., lsm
                        meta_collect.update(meta)

                    elif isinstance(meta, tuple):  # tuple[dict[str, Any]]
                        for it in meta:

                            # if isinstance(it, str):
                            #     meta_collect[]

                            for k, v in it.items():
                                if k in meta_collect:
                                    raise RuntimeError(f'{k} already existed')
                            meta_collect.update(it)
                    else:
                        fprint(f'meta type {type(meta)} parse fail', vtype='warning')

    return meta_collect
