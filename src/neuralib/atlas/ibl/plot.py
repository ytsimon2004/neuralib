from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
from matplotlib.axes import Axes

from neuralib.util.io import IBL_CACHE_DIRECTORY
from neuralib.util.util_type import PathLike, ArrayLike

__all__ = [
    #
    'IBL_PLANE_TYPE',
    'IBL_MAPPING_TYPE',
    'HEMISPHERE_TYPE',
    'IBL_BG_TYPE',
    #
    'IBLAtlasPlotWrapper'
]

IBL_PLANE_TYPE = Literal['coronal', 'sagittal', 'horizontal', 'top']
IBL_MAPPING_TYPE = Literal['Allen', 'Beryl', 'Cosmos', 'Swanson']  # merge areas
HEMISPHERE_TYPE = Literal['left', 'right', 'both']
IBL_BG_TYPE = Literal['image', 'boundary']


class IBLAtlasPlotWrapper(AllenAtlas):
    """Wrapper for iblatlas plotting usage

    .. seealso:: `<https://int-brain-lab.github.io/iblenv/atlas_examples.html>`_
    """
    regions: BrainRegions

    def __init__(self, res_um: int = 10,
                 scaling: tuple[int, int, int] = (1, 1, 1),
                 mock: bool = False,
                 source_path: PathLike | None = None,
                 alpha: float = 1.0):
        """

        :param res_um: The Atlas resolution in micrometres; one of 10, 25 or 50um
        :param scaling: Scale factor along ml, ap, dv for squeeze and stretch (default: [1, 1, 1])
        :param mock: For testing purposes, return atlas object with image comprising zeros
        :param source_path: The location of the image volume. By default, use ``IBL_CACHE_DIRECTORY``
        :param alpha:
        """

        if source_path is None:
            source_path = IBL_CACHE_DIRECTORY

        super().__init__(res_um, scaling, mock, source_path)

        self._alpha = alpha

    def get_acronym_list(self, mapping: IBL_MAPPING_TYPE) -> list[str]:
        """get acronym list"""
        # noinspection PyUnresolvedReferences
        return list(self.regions.acronym[self.regions.mappings[mapping]])

    def plot_scalar_on_slice(
            self,
            regions: ArrayLike, *,
            values: ArrayLike | None = None,
            coord: int = -570,
            plane: IBL_PLANE_TYPE = 'coronal',
            mapping: IBL_MAPPING_TYPE = 'Allen',
            hemisphere: HEMISPHERE_TYPE = 'left',
            background: IBL_BG_TYPE = 'image',
            cmap: str = 'viridis',
            clevels: ArrayLike | None = None,
            show_cbar: bool = False,
            empty_color: str = 'silver',
            ax: Axes | None = None,
            vector: bool = False,
            slice_files: np.ndarray | None = None,
            auto_merge: bool = True,
            verbose: bool = True,
            **kwargs
    ) -> tuple[plt.Figure, Axes] | tuple[plt.Figure, Axes, plt.colorbar]:
        """
        Plot the slice view.

        See detail in ``iblatlas.plots.plot_scalar_on_slice``

        .. seealso:: `<https://int-brain-lab.github.io/iblenv/notebooks_external/atlas_plotting_scalar_on_slice.html>`_

        :param regions:
        :param values:
        :param coord:
        :param plane:
        :param mapping:
        :param hemisphere:
        :param background:
        :param cmap:
        :param clevels:
        :param show_cbar:
        :param empty_color:
        :param ax:
        :param vector:
        :param slice_files:
        :param auto_merge: merge for the certain mapping
        :param verbose:
        :param kwargs:
        :return:
        """

        from iblatlas.plots import plot_scalar_on_slice

        if not isinstance(regions, np.ndarray):
            regions = np.array(regions)

        if values is None:
            values = np.arange(regions.size)

        if auto_merge:
            regions = np.unique(self.regions.acronym2acronym(regions, mapping=mapping))
            values = np.arange(regions.size)

        if verbose:
            print(f'Plot region: {list(regions)}')

        # noinspection PyTypeChecker
        return plot_scalar_on_slice(
            regions, values, coord, plane, mapping,
            hemisphere, background, cmap, clevels, show_cbar, empty_color,
            self, ax, vector, slice_files, **kwargs
        )

    def plot_points_on_slice(
            self,
            xyz: np.ndarray,
            values: ArrayLike | None = None,
            coord: int = -570,
            plane: IBL_PLANE_TYPE = 'coronal',
            mapping: IBL_MAPPING_TYPE = 'Allen',
            background: IBL_BG_TYPE = 'boundary',
            cmap: str = 'Reds',
            clevels: ArrayLike | None = None,
            show_cbar: bool = True,
            aggr: Literal['sum', 'count', 'mean', 'std', 'median', 'min', 'max'] = 'count',
            fwhm: int = 0,
            ax: Axes | None = None,
    ) -> tuple[plt.Figure, Axes] | tuple[plt.Figure, Axes, plt.colorbar]:
        """
        plot_points_on_slice

        :param xyz:
        :param values:
        :param coord:
        :param plane:
        :param mapping:
        :param background:
        :param cmap:
        :param clevels:
        :param show_cbar:
        :param aggr:
        :param fwhm:
        :param ax:
        :return:
        """
        from iblatlas.plots import plot_points_on_slice

        return plot_points_on_slice(
            xyz, values, coord, plane, mapping,
            background, cmap, clevels, show_cbar, aggr, fwhm,
            self, ax
        )

    def _plot_slice(self, im, extent, ax=None, cmap=None, volume=None, **kwargs):
        """Overwrite by pass other kwargs"""
        if ax is None:
            ax = plt.gca()
            ax.axis('equal')

        if cmap is None:
            cmap = plt.get_cmap('bone')

        if volume == 'boundary':
            imb = np.zeros((*im.shape[:2], 4), dtype=np.uint8)
            imb[im == 1] = np.array([0, 0, 0, 255])
            im = imb

        ax.imshow(im, extent=extent, cmap=cmap, alpha=self._alpha)
