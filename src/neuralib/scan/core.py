import abc
from pathlib import Path
from typing import Any, Literal

import numpy as np

from neuralib.typing import PathLike

__all__ = [
    'SceneIdx',
    'DimCode',
    'AbstractScanner',
]

SceneIdx = int
"""0-base scan position/scene"""

DimCode = str
"""
Dimension string code representing the order of axes in the data array.
Common codes (subset of tifffile):
    S - scene/series
    
    T - time
    
    C - channel
    
    Z - z plane (depth)
    
    Y - image height
    
    X - image width
    
    A - samples (for RGB/A)
    
Example: 'SZCYX'
"""


class AbstractScanner(metaclass=abc.ABCMeta):
    """Abstract Base Class for confocal microscopy image data loaders.

    Provides a common interface for accessing metadata and image data
    from various confocal file formats
    """

    _filepath: Path
    _metadata: dict[str, Any]

    def __init__(self, filepath: PathLike):
        self._filepath = Path(filepath).resolve()
        self._metadata = self._load_metadata()

    def __enter__(self):
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context, ensuring resources are released."""
        self.close()

    def close(self):
        """
        Explicitly close any open resources associated with the scanner.
        Subclasses should override this if they open files or other resources.
        """
        pass

    @property
    def filepath(self) -> Path:
        """The absolute path to the loaded confocal file."""
        return self._filepath

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the metadata dictionary parsed from the file."""
        return self._metadata

    @abc.abstractmethod
    def _load_metadata(self) -> dict[str, Any]:
        """
        Load essential metadata from the file.

        This method is called during initialization and should parse the
        minimum metadata required to fulfill the basic properties like
        n_scenes, dimensions, etc. More extensive metadata can be loaded
        lazily by ``get_full_metadata()``.

        :return: A dictionary containing essential metadata
        """
        pass

    @property
    @abc.abstractmethod
    def dimcode(self) -> DimCode:
        """Get dimension code for the image array"""
        pass

    @property
    @abc.abstractmethod
    def n_scenes(self) -> int:
        """Total number of scenes (positions/series) in the file."""
        pass

    @abc.abstractmethod
    def get_channel_names(self, scene_idx: SceneIdx) -> list[str]:
        """
        Get the names of the fluorescence channels for a specific scene.

        :param scene_idx: The 0-based index of the scene.
        :return: A list of strings representing the channel names. Returns
            generic names (e.g., ['Channel 1', 'Channel 2']) if specific
            names are not available.
        :raise IndexError: If scene_idx is out of bounds.
        """
        pass

    @abc.abstractmethod
    def view(self, **kwargs) -> np.ndarray:
        """
        view the 2D imaging array.
        :return: A NumPy image array for view. `Array[Any, [X, Y]]`
        """
        pass

    @staticmethod
    def z_projection(stacks: np.ndarray,
                     project_type: Literal['avg', 'max', 'min', 'std', 'median'] = 'max',
                     axis: int = 0) -> np.ndarray:
        """
        Computes a z-projection of a stack of images along a specified axis using
        a chosen method of projection. The function provides flexibility to select
        different projection types such as average, maximum, minimum, standard
        deviation, or median. The default projection type is `max`.

        :param stacks: A multidimensional NumPy array representing the stack of images for z-projection.
        :param project_type: The type of projection to perform. Accepted values are:
            'avg', 'max', 'min', 'std', or 'median'. The default is 'max'.
        :param axis: The axis along which the projection will be calculated.
            This is generally set according to the stack structure. Default is 0.
        :return: A NumPy array containing the result of the z-projection along the specified axis.
        :raises ValueError: If the given project_type is not one of the accepted values.
        """
        match project_type:
            case 'avg':
                return np.mean(stacks, axis=0)
            case 'max':
                return np.max(stacks, axis=0)
            case 'min':
                return np.min(stacks, axis=0)
            case 'std':
                return np.std(stacks, axis=0)
            case 'median':
                return np.median(stacks, axis=axis)
            case _:
                raise ValueError(f'unknown project type:{project_type}')
