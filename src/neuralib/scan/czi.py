from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, final, Literal
from xml.etree.ElementTree import tostring

import aicspylibczi
import numpy as np
import xmltodict

from neuralib.typing import PathLike
from neuralib.util.unstable import unstable
from .core import AbstractScanner, SceneIdx, DimCode

__all__ = ['czi_file', 'CziScanner']


@contextmanager
def czi_file(filepath: PathLike) -> Generator[CziScanner, None, None]:
    """
    context manager for load a czi file

    :param filepath: czi file path
    :return: :class:`~CziScanner`
    """
    if Path(filepath).suffix != '.czi':
        raise ValueError('czi file only')

    scanner = CziScanner(filepath)
    try:
        yield scanner
    finally:
        scanner.close()


@final
class CziScanner(AbstractScanner):
    """CZI confocal image data"""

    _czi_file: aicspylibczi.CziFile | None
    _consistent_scan_configs: bool

    def __init__(self, filepath: PathLike):
        self._czi_file = aicspylibczi.CziFile(filepath)
        super().__init__(filepath)

    def _load_metadata(self) -> dict[str, Any]:
        dim = self._czi_file.get_dims_shape()
        first_scene_dims = dim[0]
        if 'S' in first_scene_dims and len(dim) == 1:
            self._n_scenes = first_scene_dims['S'][1]
            self._consistent_scan_configs = True
        elif len(dim) > 1:
            self._n_scenes = len(dim)
            self._consistent_scan_configs = len(set(tuple(sorted(d.items())) for d in dim)) == 1
        else:
            self._n_scenes = 1
            self._consistent_scan_configs = True

        xml_string = tostring(self._czi_file.meta, encoding='utf-8').decode('utf-8')
        return xmltodict.parse(xml_string)

    def close(self):
        if self._czi_file is not None:
            self._czi_file = None

    @property
    def czi_file(self) -> aicspylibczi.CziFile:
        """get ``aicspylibczi.CziFile`` object"""
        return self._czi_file

    @property
    def consistent_config(self) -> bool:
        """Checks if the CZI file have consistent scanning configs across scenes"""
        return self._consistent_scan_configs

    @property
    def is_mosaic(self) -> bool:
        """Checks if the CZI file is marked as mosaic by the reader."""
        return self._czi_file.is_mosaic()

    @property
    def n_scenes(self) -> int:
        """Total number of scenes (positions/series) in the file."""
        return self._n_scenes

    @property
    def dimcode(self) -> DimCode:
        return self.czi_file.dims

    def get_code(self, scene_idx: SceneIdx, code: Literal['S', 'C', 'Z', 'X', 'Y', 'M']) -> int:
        """
        Retrieves a specific shape value associated with a given scene index and
        access code from the internal czi_file data structure.

        :param scene_idx: The index representing a specific scene in the czi_file's dimensions.
        :param code: A string key for accessing a particular value related to the given scene index.
        :return: Returns the shape value corresponding to the scene index and ``code``
        """
        if scene_idx < 0 or scene_idx >= self.n_scenes:
            raise ValueError(f'invalid scene index:{scene_idx}')

        dim_shape = self.czi_file.get_dims_shape()
        if self.consistent_config:
            return dim_shape[0][code][1]
        else:
            return dim_shape[scene_idx][code][1]

    def get_tile_info(self, scene_idx: SceneIdx = 0) -> dict[str, Any] | None:
        """Extracts tile region information from metadata, if available.
        Returns None if not a tiled acquisition or info is missing"""
        tile_region_info = self.metadata \
            .get('ImageDocument', {}) \
            .get('Metadata', {}) \
            .get('Experiment', {}) \
            .get('ExperimentBlocks', {}) \
            .get('AcquisitionBlock', {}) \
            .get('SubDimensionSetups', {}) \
            .get('RegionsSetup', {}) \
            .get('SampleHolder', {}) \
            .get('TileRegions', {}) \
            .get('TileRegion')

        if tile_region_info is None:
            return None

        if isinstance(tile_region_info, list):
            if 0 <= scene_idx < len(tile_region_info):
                return tile_region_info[scene_idx]  # Return specific region dict
            else:
                return None
        elif isinstance(tile_region_info, dict):
            if scene_idx == 0:
                return tile_region_info
            else:
                return tile_region_info
        else:
            return None

    def get_channel_names(self, scene_idx: SceneIdx) -> list[str]:
        """Get the names of the fluorescence channels for a specific scene.
        (Implementation copied from czi_scanner_impl_v1 for completeness)
        """
        if not 0 <= scene_idx < self.n_scenes:
            raise IndexError(f"scene_idx {scene_idx} out of bounds for {self.n_scenes} scenes.")

        info = self.metadata.get('ImageDocument', {}).get('Metadata', {})
        channels_info = info.get('Information', {}).get('Image', {}).get('Dimensions', {}).get('Channels', {}).get('Channel')
        if isinstance(channels_info, list):
            names = [ch.get('@Name', f"Channel {i}") for i, ch in enumerate(channels_info)]
        elif isinstance(channels_info, dict):
            names = [channels_info.get('@Name', "Channel 0")]
        else:
            n_channels = self.get_code(scene_idx, 'C')
            names = [f"Channel {i}" for i in range(n_channels)]
        return names

    @unstable(doc=False)
    def view(self, scene: SceneIdx | None = None,
             channel: int = 0,
             depth: int | slice | np.ndarray | None = None,
             project_type: Literal['avg', 'max', 'min', 'std', 'median'] = 'max',
             norm: bool = True) -> np.ndarray:
        """
        Generates a view of the image data based on the provided parameters such as scene,
        channel, depth, projection type, and normalization. The function retrieves image data
        from either a mosaic file or a standard image file, processes it according to the
        specified depth, performs projection if required, and applies normalization if
        requested.

        :param scene: Scene index to be used for image loading. If None and the file is non-mosaic, defaults to 0.
              Used only for non-mosaic CZI files.
        :param channel: Index of the channel to be visualized. Defaults to 0.
        :param depth: Z-plane depth index, slice, or array. If None, all Z-slices are loaded.
              Can also be a specific integer index or an array of indices/slice.
        :param project_type: Type of Z-projection to apply if multiple Z-slices are selected.
              Options include 'avg', 'max', 'min', 'std', 'median'. Defaults to 'max'.
        :param norm: Whether to normalize the image intensity values. If True, divides the
              data by its maximum intensity value. Defaults to True.
        :return: A NumPy array containing the processed image data, which may be normalized
              and/or projected based on the input parameters.
        """

        kwargs = {}
        if self.is_mosaic:
            fn = self._czi_file.read_mosaic
            nz = np.max([self.get_code(i, 'Z') for i in range(self.n_scenes)])
        else:
            fn = self._czi_file.read_image
            scene = 0 if scene is None else scene
            nz = self.get_code(scene, 'Z')
            kwargs.update({'S': scene})

        #
        if depth is None:
            depth = np.arange(nz)
        if isinstance(depth, int):
            img = fn(C=channel, Z=depth, **kwargs).squeeze()
        elif isinstance(depth, (np.ndarray, slice)):
            stacks = np.array([fn(C=channel, Z=z, **kwargs)[0].squeeze() for z in depth])  # (C, )
            img = self.z_projection(stacks, project_type)
        else:
            raise TypeError(f'unknown type {type(depth)}')

        if norm:
            img = img / np.max(img)

        return img
