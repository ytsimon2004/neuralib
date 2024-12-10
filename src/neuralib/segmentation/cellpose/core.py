import abc
from pathlib import Path
from typing import TypedDict, Literal, Union, Optional, final, Any

import attrs
import cellpose.gui.gui
import cv2
import napari
import numpy as np
from typing_extensions import Self

from neuralib.argp import argument, as_argument
from neuralib.segmentation.base import AbstractSegmentationOption
from neuralib.typing import PathLike
from neuralib.util.verbose import fprint

__all__ = ['CPOSE_MODEL',
           'AbstractSegmentationOption',
           'AbstractCellPoseOption',
           'CellPoseEvalResult']


class ChannelDict(TypedDict, total=False):
    none: int
    gray: int
    red: int
    green: int
    blue: int


CELLPOSE_CHANNEL_DICT: ChannelDict = {
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

    chan_seg: int = argument(
        '-C', '--chan',
        default=CELLPOSE_CHANNEL_DICT['gray'],
        help=f'channel for segmentation default:{CELLPOSE_CHANNEL_DICT}'
    )

    chan_nuclear: int = argument(
        '-N', '-nuclear',
        default=CELLPOSE_CHANNEL_DICT['gray'],
        help='nuclear channel'
    )

    diameter: int = argument(
        '--diameter',
        default=7,
        help='diameter for each neuron (number of each pixel)'
    )

    cellpose_view: bool = argument(
        '--cp', '--cpose',
        help='launch cellpose gui for the analyzed result'
    )

    def seg_output(self, filepath: Path) -> Path:
        return filepath.with_name(filepath.stem + '_seg').with_suffix('.npy')

    # noinspection PyTypeChecker
    def launch_napari(self):
        file = self.seg_output(self.file)
        if not file.exists() or self.force_re_eval:
            self.eval()

        res = CellPoseEvalResult.load(file)

        viewer = napari.Viewer()
        viewer.add_image(res.image, name='image', colormap='cyan')
        viewer.add_image(res.nan_masks(), name='mask', colormap='twilight_shifted', opacity=0.5)
        viewer.add_image(res.nan_outlines(), name='outline', opacity=0.5)

        napari.run()

    def launch_cellpose_gui(self):
        """AttributeError: 'MainW' object has no attribute 'load_3D'. Cellpose version 3.0.10.

        TODO open issue in cellpose -> move ``load_3D`` instance attribute to line above ``io._load_image()`` in ``MainW.__init__()``
        """

        file = self.seg_output(self.file)
        if not file.exists() or self.force_re_eval:
            self.eval()

        cellpose.gui.gui.run(image=str(self.file))  # finding seg result in the same dir


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
    """
    Cellpose results

    `Dimension parameters`:

        N = Number of segmented cell

        W = Image width

        H = Image height
    """

    # ====== #
    # Inputs #
    # ====== #

    filename: str
    """image file name"""

    image: Union[np.ndarray, list[np.ndarray]]
    """image array"""

    diameter: float
    """neuronal diameter"""

    chan_choose: list[int]
    """[chan_seg, chan_nuclear]"""

    # ======= #
    # Results #
    # ======= #

    masks: np.ndarray
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
    """outlines of ROIs. `Array[uint16, [H, W]]`"""

    ismanual: np.ndarray = attrs.field(default=np.array([]), kw_only=True)
    """whether or not mask k was manually drawn or computed by the cellpose algorithm. `Array[bool, N]`"""

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

    def save_seg_file(self, image_file: str) -> None:
        """Save as ``seg.npy`` file`"""
        from cellpose.io import masks_flows_to_seg

        if not isinstance(self.image, list):
            self.image = [self.image]

        masks_flows_to_seg(self.image,
                           self.masks,
                           self.flows,
                           image_file,
                           diams=self.diameter,
                           imgs_restore=self.img_restore,
                           restore_type=self.restore,
                           ratio=self.ratio)

    def save_roi(self, output_file: PathLike) -> None:
        """Save as imageJ ``.roi`` file.
        CHECKOUT native BUG: `<https://github.com/MouseLand/cellpose/issues/969>`_
        """
        from roifile import ImagejRoi, ROI_TYPE, ROI_OPTIONS

        points = np.fliplr(self.points)  # XY rotate in .roi format
        roi = ImagejRoi(
            roitype=ROI_TYPE.POINT,
            options=ROI_OPTIONS.PROMPT_BEFORE_DELETING | ROI_OPTIONS.SUB_PIXEL_RESOLUTION,
            n_coordinates=self.points.shape[0],
            integer_coordinates=points,
            subpixel_coordinates=points
        )

        roi.tofile(output_file)

    def nan_masks(self) -> np.ndarray:
        """value 0 in ``masks`` to nan"""
        masks = self.masks.copy().astype(np.float_)
        masks[masks == 0] = np.nan

        return masks

    def nan_outlines(self) -> np.ndarray:
        """value 0 in ``outlines`` to nan"""
        outlines = self.outlines.copy().astype(np.float_)
        outlines[outlines == 0] = np.nan

        return outlines

    @property
    def points(self) -> np.ndarray:
        """Calculate center of each segmented area in pixel. `Array[int, N]`"""
        labels = np.unique(self.masks)
        labels = labels[labels != 0]  # remove background

        n_neurons = len(labels)
        centers = np.zeros((n_neurons, 2))
        for i, label in enumerate(labels):
            segment_coords = np.argwhere(self.masks == label)
            center = segment_coords.mean(axis=0)
            centers[i] = center

        return np.round(centers).astype(int)
