from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, final

import numpy as np
import tifffile

from neuralib.plot import plot_figure
from neuralib.plot.colormap import insert_colorbar
from neuralib.scanner import (
    AbstractConfocalScanner,
    parse_tif_meta,
    SceneIdx,
    DimCode,
    ZEISS_LSM_CHANNELS_ORDER,
    ZPROJ_TYPE
)
from neuralib.util.util_type import PathLike

__all__ = ['LSMConfocalScanner']


# TODO multiple scenes not impl.
# TODO sep plot func from the class
@final
@dataclasses.dataclass
class LSMConfocalScanner(AbstractConfocalScanner):
    """lsm confocal image data"""
    lsmfile: np.ndarray
    """lsm image array"""
    meta: dict[str, Any]

    @classmethod
    def load(cls, filepath: PathLike):
        lsm = tifffile.imread(filepath)
        return LSMConfocalScanner(lsm, cls.get_meta(filepath))

    @classmethod
    def get_meta(cls, filepath: PathLike) -> dict[str, Any]:
        return parse_tif_meta(Path(filepath), is_lsm=True)

    def get_dim_code(self) -> DimCode:
        return DimCode('ZCYX')

    @property
    def width(self) -> dict[SceneIdx, int]:
        return {0: self.lsmfile.shape[3]}

    @property
    def height(self) -> dict[SceneIdx, int]:
        return {0: self.lsmfile.shape[2]}

    @property
    def n_channels(self) -> dict[SceneIdx, int]:
        return {0: self.lsmfile.shape[1]}

    @property
    def n_zstacks(self) -> dict[SceneIdx, int]:
        return {0: self.lsmfile.shape[0]}

    def get_image(self, channel: int,
                  depth: int | slice | np.ndarray | None = None,
                  zproj_type: ZPROJ_TYPE = 'max',
                  norm: bool = True) -> np.ndarray:
        """

        :param channel:
        :param depth:
        :param zproj_type:
        :param norm:
        :return:
        """

        if depth is None:
            depth = np.arange(self.lsmfile.shape[0])

        img = self.lsmfile[depth, channel, :, :]

        if img.ndim == 3 and img.shape[0] != 1:
            img = self._zproj(img, zproj_type)

        if norm:
            img = img / np.max(img)

        return img

    def imshow(self, channel: int,
               depth: int | slice | np.ndarray | None = None,
               add_scale_bar: bool = True,
               zproj_type: ZPROJ_TYPE = 'max',
               norm: bool = True,
               output: PathLike | None = None):
        """

        :param channel:
        :param depth:
        :param add_scale_bar:
        :param zproj_type:
        :param norm:
        :param output:
        :return:
        """

        img = self.get_image(channel, depth, zproj_type, norm)

        with plot_figure(output) as ax:
            im = ax.imshow(img, cmap='Greys', vmax=0.5 if norm else None)
            insert_colorbar(ax, im)

            if add_scale_bar:
                # TODO
                pass

    def plot_merge_channel(self):
        with plot_figure(None) as ax:
            for c in range(self.lsmfile.shape[1]):
                img = self.get_image(c, depth=None)
                ax.imshow(img, cmap=ZEISS_LSM_CHANNELS_ORDER[c], alpha=0.8, vmax=0.3)
