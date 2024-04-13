from __future__ import annotations

import dataclasses
from functools import cached_property
from pathlib import Path
from typing import final, NamedTuple, Literal, Any
from xml.etree.ElementTree import tostring

import aicspylibczi
import attrs
import matplotlib.pyplot as plt
import numpy as np
import xmltodict
from matplotlib.axes import Axes

from neuralib.scanner import AbstractConfocalScanner, SceneIdx, DimCode, ZEISS_CZI_CHANNELS_ORDER, ZPROJ_TYPE
from neuralib.util.util_type import PathLike
from neuralib.util.utils import joinn

__all__ = ['CziConfocalScanner']


class ImageFileInfo(NamedTuple):
    channel: int | Literal['merge']
    stack: int
    tile: int

    scene: int | None = None

    def build_filename(self, *arg, sep: str = '_') -> str:
        ret = joinn(sep, *arg)
        if self.scene is not None:
            ret += f'_s{self.scene}'
        ret += f'_c{self.channel}_z{self.stack}_t{self.tile}.tiff'
        return ret

    @property
    def channel_name(self) -> str:
        if isinstance(self.channel, int):
            return ZEISS_CZI_CHANNELS_ORDER[self.channel]
        else:
            return self.channel


# TODO sep plot func from the class
@final
@attrs.define
class CziConfocalScanner(AbstractConfocalScanner):
    """czi confocal image data"""

    czifile: aicspylibczi.CziFile
    """`aicspylibczi.CziFile`"""
    filepath: Path
    meta: dict[str, Any]
    n_scenes: int = dataclasses.field(init=False, default=1)
    consistent_scan_configs: bool = dataclasses.field(init=False)
    """whether same configs (i.e., X, Y, C ...) in different scenes"""

    def __attrs_post_init__(self):
        czi_shape = self.czifile.get_dims_shape()
        s1 = czi_shape[0]['S'][1]

        if len(czi_shape) == 1 and s1 == 1:
            self.n_scenes = 1
            self.consistent_scan_configs = True

        elif len(czi_shape) == 1 and s1 != 1:
            self.n_scenes = s1
            self.consistent_scan_configs = True
        else:
            self.n_scenes = len(czi_shape)
            self.consistent_scan_configs = False

    @classmethod
    def load(cls, file: PathLike) -> CziConfocalScanner:
        czi_file = aicspylibczi.CziFile(file)
        xml = tostring(czi_file.meta, encoding='utf-8').decode('utf-8')
        return CziConfocalScanner(
            czi_file,
            Path(file),
            xmltodict.parse(xml)
        )

    @cached_property
    def tile_info(self) -> dict[str, Any]:
        return self.meta['ImageDocument']['Metadata']['Experiment']['ExperimentBlocks']['AcquisitionBlock'][
            'SubDimensionSetups']['RegionsSetup']['SampleHolder']['TileRegions']['TileRegion']

    @property
    def tile_ncols(self) -> int:
        """how many tiles for each column"""
        return self.tile_info['Columns']

    @property
    def tile_nrows(self) -> int:
        """how many tiles for each row"""
        return self.tile_info['Rows']

    def get_dim_code(self) -> DimCode:
        return DimCode(self.czifile.dims)

    @property
    def width(self) -> dict[SceneIdx, int]:
        return {s: self._get_configs('X')[s] for s in range(self.n_scenes)}

    @property
    def height(self) -> dict[SceneIdx, int]:
        return {s: self._get_configs('Y')[s] for s in range(self.n_scenes)}

    @property
    def n_phases(self) -> dict[SceneIdx, int]:
        """how many scanning face"""
        return {s: self._get_configs('H')[s] for s in range(self.n_scenes)}

    @property
    def n_channels(self) -> dict[SceneIdx, int]:
        return {s: self._get_configs('C')[s] for s in range(self.n_scenes)}

    @property
    def n_zstacks(self) -> dict[SceneIdx, int]:
        return {s: self._get_configs('Z')[s] for s in range(self.n_scenes)}

    @property
    def n_tiles(self) -> dict[SceneIdx, int]:
        return {s: self._get_configs('M')[s] for s in range(self.n_scenes)}

    @property
    def is_mosaic(self) -> bool:
        return any(value > 1 for value in self.n_tiles.values())

    def _get_configs(self, code: str) -> list[int]:
        """get scanning configs"""
        dims = self.get_dim_code()

        if code not in dims:
            raise ValueError(f'unknown code: {code}')

        if self.consistent_scan_configs:
            size = self.czifile.size
            return [size[dims.index(code)]] * self.n_scenes
        else:
            size = [it[code] for it in self.czifile.get_dims_shape()]
            return [it[1] for it in size]

    def get_image(self,
                  channel: int,
                  scene: SceneIdx | None = None,
                  depth: int | slice | np.ndarray | None = None,
                  zproj_type: ZPROJ_TYPE = 'max',
                  norm: bool = True) -> np.ndarray:
        """
        Get the image array

        :param channel: channel index
        :param scene: scanning position
        :param depth: z stacks index, if None, use all stacks
        :param zproj_type: which z projection type, refer to fiji
        :param norm: normalization, for visualization
        :return: (Y, X)
        """

        kwargs = {}
        if self.is_mosaic:
            fn = self.czifile.read_mosaic  # tiles
        else:
            fn = self.czifile.read_image
            # TODO prob better way, seems S kw only available in non-mosaic image
            scene = 0 if scene is None else scene
            kwargs.update({'S': scene})

        #
        if depth is None:
            depth = np.arange(self.n_zstacks[scene])
        if isinstance(depth, int):
            ret = fn(C=channel, Z=depth, **kwargs).squeeze()
        elif isinstance(depth, (np.ndarray, slice)):
            stacks = np.array([fn(C=channel, Z=z, **kwargs)[0].squeeze() for z in depth])  # (C, )
            ret = self._zproj(stacks, zproj_type)
        else:
            raise TypeError(f'unknown type {type(depth)}')

        if norm:
            ret = ret / np.max(ret)

        return ret

    def get_coordinates(self) -> list[tuple[int, int]]:
        """(P, 2) with xy coordinates"""
        return [(it.x, it.y) for it in self.czifile.get_all_scene_bounding_boxes().values()]

    def imshow(self,
               scene: SceneIdx | None = None,
               add_scale_bar: bool = True,
               output: PathLike | None = None,
               position_only=False):
        """Simple plot for specific config"""
        if self.n_scenes > 1:
            self._imshow_multiple_scene(scene, position_only)
        else:
            self._imshow_single_scene(scene)

        if add_scale_bar:
            # TODO
            pass

        if output is not None:
            plt.savefig(output)
        else:
            plt.show()

    def _imshow_single_scene(self, scene: SceneIdx | None):
        fig, ax = plt.subplots()
        self._plot_merge_channel(ax, scene)

    def _imshow_multiple_scene(self, scene: SceneIdx | None,
                               position_only=False, **kwargs):
        fig, ax = plt.subplots(1, 2, figsize=(16, 4))

        if not position_only:
            self._plot_merge_channel(ax[0], scene, **kwargs)

        xy = self.get_coordinates()
        for i, (x, y) in enumerate(xy):
            ax[1].plot(x, y)
            ax[1].text(x, y, s=i)
        ax[1].invert_yaxis()

    def _plot_merge_channel(self, ax: Axes,
                            scene: SceneIdx | None,
                            **kwargs):

        scene = 0 if scene is None else scene

        for c in range(self.n_channels[scene]):
            img = self.get_image(c, scene, norm=True)
            ax.imshow(img, cmap=ZEISS_CZI_CHANNELS_ORDER[c], alpha=0.8, vmax=0.1, aspect='auto', **kwargs)

    def foreach_tif_output(self, output: PathLike | None = None,
                           combine_channels: bool = False,
                           combine_tiles: bool = True):
        """

        :param output: output directory
        :param combine_channels:
        :param combine_tiles: TODO overlap compensation?
        :return:
        """

        if output is None:
            output = self.filepath.parent / self.filepath.stem
            output.mkdir(exist_ok=True, parents=True)

        if self.is_mosaic:
            # HSTCZMYX
            images = self.czifile.read_image()[0]  # (1, 1, 1, 3, 10, 112, 512, 512)
            images = np.squeeze(images)  # (3, 10, 112, 512, 512)

            if combine_channels:
                self._foreach_output_merge_channel(images, output)
            else:
                self._foreach_output_channel_sep(images, output)

        else:
            raise NotImplementedError('')

    def _foreach_output_channel_sep(self, images: np.ndarray, output: Path):
        import tifffile

        for i, ch in enumerate(images):
            for j, z in enumerate(ch):
                for k, t in enumerate(z):
                    file = ImageFileInfo(i, j, k).build_filename(*self.filepath.stem.split('_'))

                    tifffile.imwrite(output / file, t)

    def _foreach_output_merge_channel(self, images: np.ndarray, output: Path):
        """TODO (Y, X) to (Y, X, 3)?"""
        pass


def main():
    import argparse

    ap = argparse.ArgumentParser()

    ap.add_argument('-F', '--czi',
                    metavar='FILE',
                    type=Path,
                    help='czi_file filepath',
                    dest='file')

    ap.add_argument('-O', '--output',
                    metavar='FILE',
                    type=Path,
                    default=None,
                    help='output filepath',
                    dest='output')

    ap.add_argument('-P', '--position',
                    action='store_true',
                    help='only show position fig for speeding up loading image',
                    dest='position')

    ap.add_argument('-E', '--export',
                    action='store_true',
                    help='foreach export as tiff files',
                    dest='export')

    opt = ap.parse_args()

    czi = CziConfocalScanner.load(opt.file)
    if opt.export:
        czi.foreach_tif_output(output=opt.output)
    else:
        czi.imshow(output=opt.output, position_only=opt.position)


if __name__ == '__main__':
    main()
