from __future__ import annotations

import abc
import math
from typing import Final, Union, Tuple

import attrs
import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.transforms import CompositeGenericTransform
from typing_extensions import Self

from neuralib.atlas.data import DATA_SOURCE_TYPE, load_ccf_annotation, load_ccf_template, load_allensdk_annotation
from neuralib.atlas.util import PLANE_TYPE, ALLEN_CCF_10um_BREGMA
from neuralib.imglib.factory import ImageProcFactory
from neuralib.plot import plot_figure
from neuralib.util.util_type import PathLike

__all__ = [
    'load_slice_view',
    'AbstractSliceView',
    'SlicePlane'
]


def load_slice_view(source: DATA_SOURCE_TYPE,
                    plane_type: PLANE_TYPE, *,
                    output_dir: PathLike | None = None,
                    allen_annotation_res: int = 10) -> 'AbstractSliceView':
    """
    Load the mouse brain slice view

    :param source: {'ccf_annotation', 'ccf_template', 'allensdk_annotation'}
    :param plane_type: {'coronal', 'sagittal', 'transverse'}
    :param output_dir: output directory for caching
    :param allen_annotation_res: volume resolution in um. default is 10 um
    :return: :class:`AbstractSliceView`
    """
    if source == 'ccf_annotation':
        data = load_ccf_annotation(output_dir)
        res = 10
    elif source == 'ccf_template':
        data = load_ccf_template(output_dir)
        res = 10
    elif source == 'allensdk_annotation':
        data = load_allensdk_annotation(resolution=allen_annotation_res, output_dir=output_dir)
        res = allen_annotation_res
    else:
        raise ValueError('')

    return AbstractSliceView(source, plane_type, res, data)


class AbstractSliceView(metaclass=abc.ABCMeta):
    source_type: Final[DATA_SOURCE_TYPE]
    plane_type: Final[PLANE_TYPE]
    resolution: Final[int]
    """um/pixel"""
    reference: Final[np.ndarray]
    """(AP, DV, ML)"""

    grid_x: Final[np.ndarray]
    grid_y: Final[np.ndarray]

    reference_verbose: str = ''

    def __new__(cls, source_type: DATA_SOURCE_TYPE,
                plane: PLANE_TYPE,
                resolution: int,
                reference: np.ndarray):
        if plane == 'coronal':
            return object.__new__(CoronalSliceView)
        elif plane == 'sagittal':
            return object.__new__(SagittalSliceView)
        elif plane == 'transverse':
            return object.__new__(TransverseSliceView)
        else:
            raise ValueError('')

    def __init__(self, source_type: DATA_SOURCE_TYPE,
                 plane: PLANE_TYPE,
                 resolution: int,
                 reference: np.ndarray):
        """

        :param source_type:
        :param plane:
        :param resolution:
        :param reference: (AP, DV, ML)
        """
        self.source_type = source_type
        self.plane_type = plane
        self.resolution = resolution
        self.reference = reference

        self.grid_y, self.grid_x = np.mgrid[0:self.height, 0:self.width]

        self._check_attrs()

    def _check_attrs(self):
        if self.resolution == 10:
            assert self.reference.shape == (1320, 800, 1140)
        elif self.resolution == 25:
            assert self.reference.shape == (528, 320, 456)

    @property
    def bregma(self) -> np.ndarray:
        if self.resolution == 10:
            return ALLEN_CCF_10um_BREGMA
        raise NotImplementedError('')

    @property
    def n_ap(self) -> int:
        """number of slices along AP axis"""
        return self.reference.shape[0]

    @property
    def n_dv(self) -> int:
        """number of slices along DV axis"""
        return self.reference.shape[1]

    @property
    def n_ml(self) -> int:
        """number of slices along ML axis"""
        return self.reference.shape[2]

    @property
    @abc.abstractmethod
    def n_planes(self) -> int:
        """number of planes in a specific plane view"""
        pass

    @property
    @abc.abstractmethod
    def width(self) -> int:
        """width (pixel) in a specific plane view"""
        pass

    @property
    @abc.abstractmethod
    def height(self) -> int:
        """height (pixel) in a specific plane view"""
        pass

    @property
    def width_mm(self) -> float:
        """width (um) in a specific plane view"""
        return self.width * self.resolution / 1000

    @property
    def height_mm(self) -> float:
        """height (um) in a specific plane view"""
        return self.height * self.resolution / 1000

    @property
    @abc.abstractmethod
    def reference_point(self) -> int:
        """reference point in a specific plane view. aka, bregma plane index"""
        pass

    @property
    @abc.abstractmethod
    def project_index(self) -> Tuple[int, int, int]:
        """plane(p), x, y of index order in (AP, DV, ML)

        :return: (p, x, y)
        """
        pass

    def plane_at(self, slice_index: int) -> SlicePlane:
        return SlicePlane(slice_index, int(self.width // 2), int(self.height // 2), 0, 0, self)

    def offset(self, h: int, v: int) -> np.ndarray:
        """

        :param h: horizontal plane diff to the center. right side positive.
        :param v: vertical plane diff to the center. bottom side positive.
        :return: (H, W) array
        """
        x_frame = np.round(np.linspace(-h, h, self.width)).astype(int)
        y_frame = np.round(np.linspace(-v, v, self.height)).astype(int)
        return np.add.outer(y_frame, x_frame)

    def plane(self, offset: Union[int, tuple[int, int, int], np.ndarray]) -> np.ndarray:
        """Get image plane.

        :param offset: Array[int, height, width] or tuple (plane, dh, dv)
        :return:
        """
        if isinstance(offset, int):
            offset = np.full_like((self.height, self.width), offset)
        elif isinstance(offset, tuple):
            offset = offset[0] + self.offset(offset[1], offset[2])
        elif not isinstance(offset, np.ndarray):
            raise TypeError(str(type(offset)))

        offset[offset < 0] = 0
        offset[offset > self.n_planes] = self.n_planes - 1

        return self.reference[self.coor_on(offset, (self.grid_x, self.grid_y))]

    def coor_on(self, plane: np.ndarray,
                o: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, ...]:
        """
        map slice point (x, y) at plane *plane* back to volume point (ap, dv, ml)

        :param plane: plane number of array
        :param o: tuple of (x, y)
        :return: (ap, dv, ml)
        """
        pidx, xidx, yidx = self.project_index
        ret = [0, 0, 0]
        ret[pidx] = plane
        ret[xidx] = o[0]
        ret[yidx] = o[1]

        return tuple(ret)


class CoronalSliceView(AbstractSliceView):
    reference_verbose = 'AP'

    @property
    def n_planes(self) -> int:
        return self.n_ap

    @property
    def width(self) -> int:
        return self.n_ml

    @property
    def height(self) -> int:
        return self.n_dv

    @property
    def reference_point(self) -> int:
        return self.bregma[0]

    @property
    def project_index(self) -> Tuple[int, int, int]:
        return 0, 2, 1


class SagittalSliceView(AbstractSliceView):
    reference_verbose = 'ML'

    @property
    def n_planes(self) -> int:
        return self.n_ml

    @property
    def width(self) -> int:
        return self.n_ap

    @property
    def height(self) -> int:
        return self.n_dv

    @property
    def reference_point(self) -> int:
        return self.bregma[2]

    @property
    def project_index(self) -> tuple[int, int, int]:
        return 2, 0, 1  # p=ML, x=AP, y=DV


class TransverseSliceView(AbstractSliceView):
    reference_verbose = 'DV'

    @property
    def n_planes(self) -> int:
        return self.n_dv

    @property
    def width(self) -> int:
        return self.n_ml

    @property
    def height(self) -> int:
        return self.n_ap

    @property
    def reference_point(self) -> int:
        return self.bregma[1]

    @property
    def project_index(self) -> tuple[int, int, int]:
        return 1, 2, 0


@attrs.define
class SlicePlane:
    """2D Wrapper class of *SliceView for specific plane"""
    slice_index: int  # anchor index
    ax: int  # anchor x
    ay: int  # anchor y
    dw: int  # in um
    dh: int
    view: AbstractSliceView

    @property
    def image(self) -> np.ndarray:
        return self.view.plane(self.plane_offset)

    @property
    def plane_offset(self) -> np.ndarray:
        offset = self.view.offset(self.dw, self.dh)
        return self.slice_index + offset - offset[self.ay, self.ax]

    @property
    def reference_value(self) -> float:
        """relative to reference point"""
        factor = 1000 / self.view.resolution
        return round((self.view.reference_point - self.slice_index) / factor, 2)

    def with_offset(self, dw: int, dh: int) -> Self:
        return attrs.evolve(self, dw=dw, dh=dh)

    def with_angle_offset(self, deg_x: float, deg_y: float) -> Self:
        """

        :param deg_x:
        :param deg_y:
        :return:
        """
        rx = np.deg2rad(deg_x)
        ry = np.deg2rad(deg_y)

        dw = int(-self.view.width * math.tan(rx) / 2)
        dh = int(self.view.height * math.tan(ry) / 2)

        return self.with_offset(dw, dh)

    def plot(self, ax: Axes | None = None,
             to_um: bool = True,
             with_annotation: bool = False,
             cbar: bool = False,
             with_title: bool = False,
             affine_transform: bool = False,
             customized_trans: bool = False,
             **kwargs) -> tuple[AxesImage, AxesImage | None, CompositeGenericTransform]:
        """
        Plot slice view

        :param ax:
        :param to_um: whether plot axis as mm or um
        :param with_annotation: if plot annotation
        :param cbar:
        :param with_title:
        :param affine_transform: whether do the affine transformation
        :param customized_trans: whether use customized transformation method. NOTE the Shear transform matrix [2, 0]
                        value might not be able to apply in other plot (shifting)
        :param kwargs: pass through the image imshow args
        :return:
        """

        if ax is None:
            with plot_figure(None) as ax:
                ret = self._plot(ax, to_um, with_annotation, cbar, with_title, affine_transform,
                                 customized_trans, **kwargs)
        else:
            ret = self._plot(ax, to_um, with_annotation, cbar, with_title, affine_transform, customized_trans, **kwargs)

        return ret

    unit: str = None

    def _get_xy_range(self, to_um=True) -> tuple[float, float, float, float]:
        if to_um:
            x0 = -self.view.width_mm / 2 * 1000
            x1 = self.view.width_mm / 2 * 1000
            y0 = self.view.height_mm * 1000
            y1 = 0
            self.unit = 'um'
        else:
            x0 = -self.view.width_mm / 2
            x1 = self.view.width_mm / 2
            y0 = self.view.height
            y1 = 0
            self.unit = 'mm'

        return x0, x1, y0, y1

    def _customized_affine_transform(self) -> np.ndarray:
        # translation
        Y = np.array([[1, 0, 0], [0, 1, -4000], [0, 0, 1]])

        # shear
        tt = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-0.03 / self.view.width, 0, 1]
        ])

        # translation
        _Y = np.array([[1, 0, 0], [0, 1, 3500], [0, 0, 1]])

        return Y @ tt @ _Y

    def _plot(self, ax: Axes,
              to_um,
              with_annotation,
              cbar,
              with_title,
              affine_transform,
              customized_trans,
              **kwargs) -> tuple[AxesImage, AxesImage | None, CompositeGenericTransform]:

        x0, x1, y0, y1 = self._get_xy_range(to_um)

        #
        if affine_transform:
            import matplotlib.transforms as mtransforms
            #
            if customized_trans:
                aff = mtransforms.Affine2D(self._customized_affine_transform())
            else:
                aff = mtransforms.Affine2D().skew_deg(-20, 0)

            aff_trans = aff + ax.transData
        else:
            aff_trans = ax.transData

        #
        image = self.image.astype(float)
        image[image <= 10] = np.nan
        im_view = ax.imshow(image, cmap='Greys', extent=(x0, x1, y0, y1), clip_on=False, transform=aff_trans, **kwargs)

        #
        if with_annotation:
            ann_img = load_slice_view('ccf_annotation',
                                      self.view.plane_type,
                                      allen_annotation_res=self.view.resolution).plane(self.plane_offset)
            ann = ImageProcFactory(ann_img).covert_grey_scale().edge_detection(10, 0).image

            ann = ann.astype(float)
            ann[ann <= 10] = np.nan

            im_ann = ax.imshow(ann, cmap='binary', extent=(x0, x1, y0, y1), alpha=0.3, clip_on=False,
                               interpolation='none', vmin=0, vmax=255, transform=aff_trans)

        else:
            im_ann = None

        #
        if cbar:
            ax.figure.colorbar(im_view)
        #
        if with_title:
            ax.set_title(f'{self.view.reference_verbose}: {self.reference_value} mm')

        ax.set(xlabel=self.unit, ylabel=self.unit)

        return im_view, im_ann, aff_trans
