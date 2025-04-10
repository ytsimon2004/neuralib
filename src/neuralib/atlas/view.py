from __future__ import annotations

import abc
import math
import warnings
from typing import Final, ClassVar, Literal, get_args

import attrs
import matplotlib.pyplot as plt
import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from typing_extensions import Self

from neuralib.atlas.data import ATLAS_NAME
from neuralib.atlas.typing import PLANE_TYPE
from neuralib.atlas.util import ALLEN_CCF_10um_BREGMA
from neuralib.util.deprecation import deprecated_func

__all__ = [
    'VIEW_TYPE',
    'get_slice_view',
    'load_slice_view',
    'AbstractSliceView',
    'SlicePlane'
]

VIEW_TYPE = Literal['annotation', 'reference']
"""View type for the slice"""


def get_slice_view(view: VIEW_TYPE,
                   plane_type: PLANE_TYPE,
                   *,
                   name: str = 'allen_mouse',
                   resolution: int = 10,
                   check_latest: bool = True) -> AbstractSliceView:
    """
    Load the mouse brain slice view

    .. seealso::

        `brainglobe-atlasap doc <https://brainglobe.info/documentation/brainglobe-atlasapi/usage/atlas-details.html>`_

    :param view: ``VIEW_TYPE``.
    :param plane_type: ``PLANE_TYPE``. {'coronal', 'sagittal', 'transverse'}
    :param name: Name of the atlas.
    :param resolution: Volume resolution in um. default is 10 um
    :param check_latest: If True, check the latest version of brain
    :return: :class:`AbstractSliceView`
    """
    match view:
        case 'annotation' | 'reference':
            atlas_name = f'{name}_{resolution}um'
            if atlas_name not in get_args(ATLAS_NAME):
                raise ValueError(f'{atlas_name} not found or not implemented')
            data = getattr(BrainGlobeAtlas(atlas_name, check_latest=check_latest), view)
        case _:
            raise ValueError(f'Unknown view: {view}')

    return AbstractSliceView(view, plane_type, resolution, data)


class AbstractSliceView(metaclass=abc.ABCMeta):
    """
    SliceView ABC for different `plane type`

    `Dimension parameters`:

        AP = anterior-posterior

        DV = dorsal-ventral

        ML = medial-lateral

        W = view width

        H = view height
    """
    REFERENCE_FROM: ClassVar[str] = ''
    """reference from which axis"""

    view_type: Final[VIEW_TYPE]
    """``VIEW_TYPE``. {'annotation', 'reference'}"""

    plane_type: Final[PLANE_TYPE]
    """`PLANE_TYPE``. {'coronal', 'sagittal', 'transverse'}"""

    resolution: Final[int]
    """um/pixel"""

    reference: Final[np.ndarray]
    """Array[float, [AP, DV, ML]]"""

    grid_x: Final[np.ndarray]
    """Array[int, [W, H]]"""

    grid_y: Final[np.ndarray]
    """Array[int, [W, H]]"""

    def __new__(cls, view_type: VIEW_TYPE,
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
            raise ValueError(f'invalid plane: {plane}')

    def __init__(self, view_type: VIEW_TYPE,
                 plane: PLANE_TYPE,
                 resolution: int,
                 reference: np.ndarray):
        """

        :param view_type: ``DATA_SOURCE_TYPE``. {'ccf_annotation', 'ccf_template', 'allensdk_annotation'}
        :param plane: `PLANE_TYPE``. {'coronal', 'sagittal', 'transverse'}
        :param resolution: um/pixel
        :param reference: Array[uint16, [AP, DV, ML]]
        """
        self.view_type = view_type
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
    def project_index(self) -> tuple[int, int, int]:
        """plane(p), x, y of index order in (AP, DV, ML)

        :return: (p, x, y)
        """
        pass

    @property
    @abc.abstractmethod
    def max_projection_axis(self) -> tuple[int, int, int]:
        pass

    def plot_max_projection(self, ax: Axes, *,
                            annotation_regions: str | list[str] | None = None,
                            annotation_cmap: str = 'hsv'):
        """
        Plot max projection for the given ``plane_type``

        :param ax: ``Axes``
        :param annotation_regions: annotation_regions
        :param annotation_cmap: camp for the annotation regions, defaults to 'hsv'
        """

        img = self.reference.max(axis=self.max_projection_axis)
        if isinstance(self, SagittalSliceView):
            img = img.T

        ext = _get_xy_range(self, to_um=True)
        ax.imshow(img, cmap='Greys', extent=ext)

        if annotation_regions is not None:
            from neuralib.atlas.data import get_leaf_in_annotation
            annotation = get_slice_view('annotation', self.plane_type, resolution=self.resolution).reference

            if isinstance(annotation_regions, str):
                annotation_regions = [annotation_regions]

            #
            region_colors = plt.get_cmap(annotation_cmap, len(annotation_regions))
            for i, r in enumerate(annotation_regions):
                ids = get_leaf_in_annotation(r, name=False)
                mask = np.isin(annotation, ids)

                region_mask = np.full(annotation.shape, np.nan, dtype=np.float64)
                region_mask[mask] = 1.0

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    proj = np.nanmax(region_mask, axis=self.max_projection_axis)

                if isinstance(self, SagittalSliceView):
                    proj = proj.T

                masked = np.ma.masked_invalid(proj)
                ax.imshow(
                    masked,
                    cmap=ListedColormap(region_colors(i)),
                    extent=ext,
                    alpha=0.7,
                    zorder=2 + i,
                )

            #
            ax.set(xlabel='um', ylabel='um')
            legend_elements = [
                Patch(facecolor=region_colors(i), label=region, alpha=0.7)
                for i, region in enumerate(annotation_regions)
            ]
            ax.legend(handles=legend_elements, title="Regions", loc='upper right')

    def plane_at(self, slice_index: int) -> 'SlicePlane':
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

    def plane(self, offset: int | tuple[int, int, int] | np.ndarray) -> np.ndarray:
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
    REFERENCE_FROM: ClassVar[str] = 'AP'

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
        return int(self.bregma[0])

    @property
    def project_index(self) -> tuple[int, int, int]:
        return 0, 2, 1

    @property
    def max_projection_axis(self) -> int:
        return 0


class SagittalSliceView(AbstractSliceView):
    REFERENCE_FROM: ClassVar[str] = 'ML'

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
        return int(self.bregma[2])

    @property
    def project_index(self) -> tuple[int, int, int]:
        return 2, 0, 1  # p=ML, x=AP, y=DV

    @property
    def max_projection_axis(self) -> int:
        return 2


class TransverseSliceView(AbstractSliceView):
    REFERENCE_FROM: ClassVar[str] = 'DV'

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
        return int(self.bregma[1])

    @property
    def project_index(self) -> tuple[int, int, int]:
        return 1, 2, 0

    @property
    def max_projection_axis(self):
        return 1


@attrs.define
class SlicePlane:
    """2D Wrapper for a specific plane"""

    slice_index: int
    """anchor index"""

    ax: int
    """anchor x"""

    ay: int
    """anchor y"""

    dw: int
    """dw in um"""

    dh: int
    """dh in um"""

    slice_view: AbstractSliceView
    """``AbstractSliceView``"""

    unit: str = 'a.u.'

    @property
    def image(self) -> np.ndarray:
        return self.slice_view.plane(self.plane_offset)

    @property
    def plane_offset(self) -> np.ndarray:
        offset = self.slice_view.offset(self.dw, self.dh)
        return self.slice_index + offset - offset[self.ay, self.ax]

    @property
    def reference_value(self) -> float:
        """relative to reference point"""
        factor = 1000 / self.slice_view.resolution
        return round((self.slice_view.reference_point - self.slice_index) / factor, 2)

    def with_offset(self, dw: int, dh: int, debug: bool = False) -> Self:
        if debug:
            deg_x, deg_y = self._value_to_angle(dw, dh)
            print(f'{dw=}, {dh=}')
            print(f'{deg_x=}, {deg_y=}')

        return attrs.evolve(self, dw=dw, dh=dh)

    def _value_to_angle(self, dw: int, dh: int) -> tuple[float, float]:
        """delta value to degree"""
        rx = math.atan(2 * dw / self.slice_view.width)
        ry = math.atan(2 * dh / self.slice_view.height)
        deg_x = np.rad2deg(rx)
        deg_y = np.rad2deg(ry)

        return deg_x, deg_y

    def with_angle_offset(self, deg_x: float = 0, deg_y: float = 0) -> Self:
        """
        with degree offset

        :param deg_x: degree in x axis (width)
        :param deg_y: degree in y axis (height)
        :return:
        """
        rx = np.deg2rad(deg_x)
        ry = np.deg2rad(deg_y)

        dw = int(self.slice_view.width * math.tan(rx) / 2)
        dh = int(self.slice_view.height * math.tan(ry) / 2)

        return self.with_offset(dw, dh)

    def plot(self,
             ax: Axes | None = None,
             to_um: bool = True,
             annotation_region: str | list[str] | None = None,
             boundaries: bool = False,
             with_title: bool = False,
             extent: tuple[float, float, float, float] | None = None,
             reference_bg_value: float | None = None,
             annotation_cmap: str = 'berlin',
             annotation_rescale: bool = True,
             **kwargs) -> None:
        """
        :param ax: The Axes object on which to plot. If None, a new figure and axes are created.
        :param to_um: A boolean flag indicating whether the coordinates should be converted to micrometers. Defaults to True.
            Only applicable if ``extent`` is None.
        :param annotation_region: The annotation region on which to plot. Defaults to None.
        :param boundaries: A boolean indicating whether to include annotations in the plot.
        :param with_title: A boolean indicating whether to include a title in the plot.
        :param extent: A tuple defining the image boundaries (left, right, bottom, top). If None, boundaries are computed internally.
        :param reference_bg_value: If specified, remove background of its value in when view_type is **reference** (i.e., set as 10).
        :param annotation_cmap: Cmap for the annotation regions if specified
        :param annotation_rescale: Rescale the image when view_type is **annotation**.
        :param kwargs: Additional keyword arguments passed to ``ax.imshow()``.
        """

        if ax is None:
            _, ax = plt.subplots()

        if extent is None:
            extent = self._get_xy_range(to_um)

        # value modify
        image = self.image.astype(float)
        if reference_bg_value is not None and self.slice_view.view_type == 'reference':
            image[image < reference_bg_value] = np.nan

        if annotation_rescale and self.slice_view.view_type == 'annotation':
            image[image == 0] = np.nan
            valid = ~np.isnan(image)
            unique_vals = np.unique(image[valid])
            remapped_image = np.full_like(image, np.nan)
            remapped_image[valid] = np.searchsorted(unique_vals, image[valid]) + 1
            image = remapped_image

        # image
        ax.imshow(image, cmap='Greys', extent=extent, clip_on=False, **kwargs)

        # annotation region
        if annotation_region is not None:
            self._plot_annotation_regions(ax, annotation_region, extent, annotation_cmap, **kwargs)

        # with boundaries
        if boundaries:
            self._plot_boundaries(ax, extent, **kwargs)

        #
        if with_title:
            ax.set_title(f'{self.slice_view.REFERENCE_FROM}: {self.reference_value} mm')

        ax.set(xlabel=self.unit, ylabel=self.unit)

    def _plot_annotation_regions(self, ax, regions, extent, cmap='berlin', **kwargs):
        from matplotlib.patches import Patch
        from neuralib.atlas.data import get_leaf_in_annotation

        annotation = (
            get_slice_view('annotation', self.slice_view.plane_type, resolution=self.slice_view.resolution)
            .plane(self.plane_offset)
            .astype(float)
        )

        if isinstance(regions, str):
            regions = [regions]

        area = np.full_like(annotation, np.nan)
        for i, r in enumerate(regions):
            ids = get_leaf_in_annotation(r, name=False)
            mx = np.isin(annotation, ids)
            area[mx] = i + 1.0

        cmap = plt.get_cmap(cmap, len(regions))

        ax.imshow(area, cmap=cmap, extent=extent, alpha=0.9, clip_on=False, **kwargs)
        legend_elements = [
            Patch(facecolor=cmap(i / len(regions)), label=region)
            for i, region in enumerate(regions)
        ]
        ax.legend(handles=legend_elements, title="Regions", loc='upper right')

    def _plot_boundaries(self, ax, extent, cmap='binary', alpha=0.3, **kwargs):
        """
        Plot the annotation boundaries

        :param ax: ``Axes``
        :param extent: A tuple defining the image boundaries (left, right, bottom, top). If None, boundaries are computed internally.
        :param cmap: Colormap to be used for the annotation image. Defaults to 'binary'.
        :param alpha: The imshow alpha, between 0 (transparent) and 1 (opaque). Defaults to 0.3.
        """
        from neuralib.imglib.array import image_array

        ann_img = (
            get_slice_view('annotation', self.slice_view.plane_type, resolution=self.slice_view.resolution)
            .plane(self.plane_offset)
        )

        ann = (
            image_array(ann_img)
            .to_gray()
            .canny_filter(10, 0)
        )

        ann = ann.astype(float)
        ann[ann <= 10] = np.nan
        ax.imshow(ann, cmap=cmap, extent=extent, alpha=alpha, clip_on=False, interpolation='none', vmin=0, vmax=255, **kwargs)

    def _get_xy_range(self, to_um: bool = True) -> tuple[float, float, float, float]:
        self.unit = 'um' if to_um else 'mm'
        return _get_xy_range(self.slice_view, to_um=to_um)


def _get_xy_range(view: AbstractSliceView, to_um: bool = True) -> tuple[float, float, float, float]:
    if to_um:
        x0 = -view.width_mm / 2 * 1000
        x1 = view.width_mm / 2 * 1000
        y0 = view.height_mm * 1000
        y1 = 0
    else:
        x0 = view.width_mm / 2
        x1 = view.width_mm / 2
        y0 = view.height
        y1 = 0

    return x0, x1, y0, y1


@deprecated_func(new='get_slice_view()', removal_version='0.4.5')
def load_slice_view(*args, **kwargs):
    return get_slice_view(*args, **kwargs)
