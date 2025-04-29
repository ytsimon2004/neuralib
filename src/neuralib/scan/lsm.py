from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, final, Generator, Literal

import numpy as np
import tifffile

from neuralib.typing import PathLike
from neuralib.util.verbose import fprint
from .core import AbstractScanner, DimCode

__all__ = ['lsm_file', 'TiffScanner']


@contextmanager
def lsm_file(filepath: PathLike) -> Generator[TiffScanner, None, None]:
    """context manager for load a lsm file

    :param filepath: lsm file path
    :return: :class:`~TiffScanner`
    """
    scanner = TiffScanner(filepath)
    try:
        yield scanner
    finally:
        scanner.close()


@final
class TiffScanner(AbstractScanner):
    """LSM confocal image data.

    **TODO**: multi-scene and different dimension data are lacking of testing
    """

    _image: np.ndarray | None
    _file_type: str

    def __init__(self, filepath: PathLike):
        self._image = tifffile.imread(filepath)
        self._file_type = Path(filepath).suffix

        super().__init__(filepath)

    def close(self):
        if self._image is not None:
            self._image = None

    def _load_metadata(self) -> dict[str, Any]:
        return parse_tif_meta(self._filepath, is_lsm=self.file_type == '.lsm')

    @property
    def image(self) -> np.ndarray:
        """image array"""
        return self._image

    @property
    def file_type(self) -> str:
        """file type of the image"""
        return self._file_type

    @property
    def dimcode(self) -> DimCode:
        match self.file_type:
            case '.lsm':
                if self.n_scenes == 1:
                    return 'ZCYX'
                else:
                    raise NotImplementedError('multi-scene not support yet')
            case _:
                raise NotImplementedError(f'{self.file_type} not support yet')

    @property
    def n_scenes(self) -> int:
        return self.metadata.get('DimensionP', 1)

    def get_channel_names(self, scene_idx=None) -> list[str]:
        return self.metadata.get('ChannelColors', {}).get('ColorNames', [])

    def view(self, channel: int = 0,
             depth: int | slice | np.ndarray | None = None,
             project_type: Literal['avg', 'max', 'min', 'std', 'median'] = 'max',
             norm: bool = True) -> np.ndarray:
        """
        Generates a view of the image data based on the provided parameters.
        Only provide a single scene view for now.

        :param channel: The channel index to select from the image. Defaults to 0.
        :param depth: Depth levels to process, which can be an integer, a slice,
            or a NumPy array. If None, all depth levels are used.
        :param project_type: Type of Z-projection to apply if multiple Z-slices are selected.
             Options include 'avg', 'max', 'min', 'std', 'median'. Defaults to 'max'.
        :param norm: A flag indicating whether to normalize the projected image by its
            maximum intensity value. Defaults to True.
        :return: A NumPy array representing the resulting image after applying depth
            projection, channel selection, and optional normalization.
        """

        if depth is None:
            depth = np.arange(self.image.shape[0])

        img = self.image[depth, channel, :, :]

        if isinstance(depth, int):
            pass
        elif isinstance(depth, (np.ndarray, slice)):
            img = self.z_projection(img, project_type)
        else:
            raise TypeError(f'unknown type {type(depth)}')

        if norm:
            img = img / np.max(img)

        return img


def parse_tif_meta(file: Path, **kwargs) -> dict[str, Any]:
    """
    Extracts metadata dict from a .tif/.tiff/.lsm file.
    Raises ValueError if the extension isn’t supported.

    :param file: Path to the file to be parsed.
    """
    suffix = file.suffix.lower()
    if suffix not in {'.tif', '.tiff', '.lsm'}:
        raise ValueError(f"Unsupported extension: {file.suffix!r}")

    meta_collect = {}
    with tifffile.TiffFile(file, **kwargs) as tif:
        # LSM metadata
        lsm = getattr(tif, 'lsm_metadata', None)
        if isinstance(lsm, dict):
            meta_collect.update(lsm)

        # ImageJ metadata
        ij = getattr(tif, 'imagej_metadata', None)
        if isinstance(ij, dict):
            meta_collect.update(ij)

        # Any other tuple of dicts you care about
        for attr in ('ome_metadata', 'ome_xml',):
            md = getattr(tif, attr, None)
            if isinstance(md, tuple):
                for part in md:
                    if not isinstance(part, dict):
                        fprint("Skipping non‐dict in %s: %r", attr, part)
                        continue
                    dup = meta_collect.keys() & part.keys()
                    if dup:
                        raise RuntimeError(f"Duplicate keys in {attr}: {dup}")
                    meta_collect.update(part)

    return meta_collect
