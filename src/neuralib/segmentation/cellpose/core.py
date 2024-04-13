import abc
from pathlib import Path
from typing import TypedDict, Literal, Union, Optional, final, Any

import attrs
import cv2
import napari
import numpy as np
from typing_extensions import Self

from neuralib.argp import argument, as_argument
from neuralib.segmentation.base import AbstractSegmentationOption
from neuralib.util.util_type import PathLike
from neuralib.util.util_verbose import fprint

__all__ = ['CPOSE_MODEL',
           'AbstractSegmentationOption',
           'AbstractCellPoseOption',
           'CellPoseEvalResult']


class ChannelDict(TypedDict):
    none: int
    gray: int
    red: int
    green: int
    blue: int


DEFAULT_CHANNEL_DICT: ChannelDict = {
    'none': -1,
    'gray': 0,
    'red': 1,
    'green': 2,
    'blue': 3
}

CPOSE_MODEL = Literal['cyto', 'cyto2', 'cyto3']


class AbstractCellPoseOption(AbstractSegmentationOption, metaclass=abc.ABCMeta):
    DESCRIPTION = 'ABC for GUI Cellpose'

    model: CPOSE_MODEL = as_argument(AbstractSegmentationOption.model).with_options(default='cyto3')

    chan_seg: str = argument(
        '-C', '--chan',
        help=f'channel for segmentation default:{DEFAULT_CHANNEL_DICT}'
    )

    chan_nuclear: str = argument(
        '-N', '--chan-nuclear',
        default=DEFAULT_CHANNEL_DICT['blue'],
        help='nuclear channel'
    )

    diameter: int = argument(
        '-D', '--diameter',
        default=7,
        help='diameter for each neuron (number of each pixel)'
    )

    cellpose_gui: bool = argument(
        '--cpose-gui',
        help='launch_cellpose_gui for the analyzed result'
    )

    @property
    def seg_result(self) -> Path:
        if not self.file.is_file():
            raise ValueError(f'{self.file} must be a file')
        return self.file.with_name(self.file.stem + '_seg').with_suffix('.npy')

    @abc.abstractmethod
    def eval(self) -> Optional['CellPoseEvalResult']:
        pass

    # noinspection PyTypeChecker
    def launch_napari(self):
        res = CellPoseEvalResult.load(self.seg_result)

        viewer = napari.Viewer()
        viewer.add_image(res.image, name='raw', colormap='cyan')
        viewer.add_image(res.masks, name='mask', opacity=0.5, colormap='red')
        viewer.add_image(res.outlines, name='outline', opacity=0.5)

        napari.run()

    def launch_cellpose_gui(self):
        """FIXME AttributeError: 'MainW' object has no attribute 'load_3D'. Cellpose version 3.0.5
        """
        from cellpose.gui.gui import run
        run(str(self.seg_result))


# ========== #
# EvalResult #
# ========== #
# TODO check doc and type in up-to date version
# TODO 3D image need to be tested

class NormParams(TypedDict):
    lowhigh: Optional[Any]
    percentile: list[float, float]
    normalize: bool
    norm3D: bool
    sharpen_radius: float
    smooth_radius: float
    tile_norm_blocksize: float
    tile_norm_smooth3D: float
    invert: bool


@final
@attrs.define
class CellPoseEvalResult:
    # inputs
    filename: str
    """image file name"""
    image: Union[np.ndarray, list[np.ndarray]]
    diameter: float
    """neuronal diameter"""
    chan_choose: list[int]
    """[chan_neg, chan_nuclear]"""

    # result
    masks: list[np.ndarray] = attrs.Factory(list)
    """each pixel in the image is assigned to an ROI (H, W)
    list of 2D arrays, labelled image, where 0=no masks; 1,2,...=mask labels
    """

    flows: list[np.ndarray] = attrs.Factory(list)
    """
    flows[0] is XY flow in RGB, 
    flows[1] is the cell probability in range 0-255 instead of 0.0 to 1.0, 
    flows[2] is Z flow in range 0-255 (if it exists, otherwise zeros),
    flows[3] is [dY, dX, cellprob] (or [dZ, dY, dX, cellprob] for 3D),
    flows[4] is pixel destinations (for internal use)
    """

    styles: list[np.ndarray] = attrs.Factory(list)
    """list of 1D arrays of length 256, 
    style vector summarizing each image, also used to estimate size of objects in image"""

    # ==================== #
    # Optional & Overwrite #
    # ==================== #

    # GUI dependent
    colors: Optional[np.ndarray] = attrs.field(default=None, kw_only=True)
    """colors for ROIs (N, 3)"""

    manual_changes: Optional[list[Any]] = attrs.field(default=None, kw_only=True)

    # CLI dependent
    est_diam: Optional[float] = attrs.field(default=None, kw_only=True)
    """estimated diameter (if run on command line)"""

    #
    model_path: int = attrs.field(default=0, kw_only=True)
    flow_threshold: Optional[float] = attrs.field(default=None, kw_only=True)
    cellprob_threshold: float = attrs.field(default=0, kw_only=True)
    normalize_params: Optional[NormParams] = attrs.field(default=None, kw_only=True)

    # restore
    img_restore: Optional[list[np.ndarray]] = attrs.field(default=None, kw_only=True)
    restore: Optional[str] = attrs.field(default=None, kw_only=True)
    ratio: float = attrs.field(default=1.0, kw_only=True)

    # Overwrite while calling save
    outlines: np.ndarray = attrs.field(default=np.array([]), kw_only=True)
    """outlines of ROIs (H, W)"""
    ismanual: np.ndarray = attrs.field(default=np.array([]), kw_only=True)
    """whether or not mask k was manually drawn or computed by the cellpose algorithm. bool array (N, )"""

    @classmethod
    def load(cls, seg_file: PathLike) -> Self:
        if not isinstance(seg_file, Path):
            seg_file = Path(seg_file)

        res = np.load(seg_file, allow_pickle=True).item()

        image_file = Path(res['filename'])
        if not image_file.exists():
            fprint(f'No image data found {image_file}', vtype='warning')
            image = None
        else:
            image = cv2.imread(res['filename'])

        return cls(**res, image=image)

    def save_seg_file(self) -> None:
        from cellpose.io import masks_flows_to_seg

        if not isinstance(self.image, list):
            self.image = [self.image]

        masks_flows_to_seg(self.image,
                           self.masks,
                           self.flows,
                           self.filename,
                           diams=self.diameter,
                           imgs_restore=self.img_restore,
                           restore_type=self.restore,
                           ratio=self.ratio)
