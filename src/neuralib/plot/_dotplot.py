from typing import Literal, ClassVar

import matplotlib.colorbar
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from neuralib.plot import plot_figure, ax_merge
from neuralib.typing import ArrayLike
from neuralib.typing import PathLike

__all__ = ['DotPlot']


class DotPlot:
    DEFAULT_COLOR_TITLE: ClassVar[str] = 'value'
    DEFAULT_SIZE_TITLE: ClassVar[str] = 'value'
    DEFAULT_MAX_MARKER_SIZE: ClassVar[float] = 500.0
    DEFAULT_MARKER_LEGEND_NUM: ClassVar[int] = 5

    def __init__(
            self,
            xlabel: ArrayLike,
            ylabel: ArrayLike,
            values: np.ndarray,
            *,
            scale: Literal['area', 'radius'] = 'radius',
            max_marker_size: float | None = None,
            size_title: str | None = None,
            size_legend_num: int | None = None,
            size_legend_as_int: bool = True,
            with_color: bool = False,
            cmap: mcolors.Colormap = 'Reds',
            colorbar_title: str | None = None,
            norm: mcolors.Normalize | None = None,
            cbar_vmin: float | None = None,
            cbar_vmax: float | None = None,
            figure_title: str | None = None,
            figure_output: PathLike | None = None,
            ax: Axes | None = None
    ):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.values = values

        self._check_instance()

        # size
        self.scale = scale
        self.max_marker_size = max_marker_size or DotPlot.DEFAULT_MAX_MARKER_SIZE
        self.size_title = size_title or DotPlot.DEFAULT_SIZE_TITLE
        self.size_legend_num = size_legend_num or DotPlot.DEFAULT_MARKER_LEGEND_NUM
        self.size_as_int = size_legend_as_int

        # color
        self.with_color = with_color
        self.colorbar_title = colorbar_title or DotPlot.DEFAULT_COLOR_TITLE
        self.cmap = cmap
        self.cbar_vmin = cbar_vmin or np.min(values)
        self.cbar_vmax = cbar_vmax or np.max(values)
        self.norm = norm or self._default_norm()

        # figure
        self.figure_title = figure_title
        self.figure_output = figure_output
        self.ax = ax

    def _check_instance(self):
        x = np.array(self.xlabel)
        y = np.array(self.ylabel)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError('xlabel and ylabel must to be 1d')

        if not np.issubdtype(x.dtype, np.str_) or not np.issubdtype(y.dtype, np.str_):
            raise TypeError('xlabel and ylabel must be string array')

        nx = len(self.xlabel)
        ny = len(self.ylabel)

        if self.values.shape != (nx, ny):
            raise ValueError('shape inconsistent')

    def _default_norm(self) -> mcolors.Normalize:
        """default cbar norm"""
        return mcolors.Normalize(vmin=self.cbar_vmin, vmax=self.cbar_vmax)

    @property
    def scaling_factor(self) -> float:
        max_value = np.max(self.values)
        if self.scale == 'radius':
            max_value **= 2
        return self.max_marker_size / max_value

    @property
    def size(self) -> np.ndarray:
        """map value to size, which maximal ``MAX_SCATTER_SIZE``"""
        values = np.array([self.values]).flatten()
        if self.scale == 'radius':
            return (values ** 2) * self.scaling_factor
        elif self.scale == 'area':
            return values * self.scaling_factor
        else:
            raise ValueError(f'{self.scale}')

    def plot(self, **kwargs):
        values = np.array([self.values]).flatten()
        x, y = np.meshgrid(self.xlabel, self.ylabel, indexing='ij')

        if self.ax is None:
            self._plot_figure(x, y, values, **kwargs)
        else:
            self._plot_ax(x, y, values, **kwargs)

    def _plot_figure(self, x, y, values, **kwargs):
        """Plot in figure level"""
        with plot_figure(self.figure_output, 7, 4, figsize=(8, 5)) as _ax:
            ax = ax_merge(_ax)[:, :3]
            if self.with_color:
                ax.scatter(x.ravel(), y.ravel(),
                           s=self.size,
                           c=values,
                           cmap=self.cmap,
                           clip_on=False,
                           norm=self.norm,
                           edgecolor='gray',
                           **kwargs)

                ax_cbar = _ax[6, 3]
                self._plot_colorbar(ax_cbar, self.norm)
            else:
                ax.scatter(x.ravel(), y.ravel(),
                           s=self.size,
                           c='k',
                           clip_on=False,
                           **kwargs)
                ax_cbar = _ax[6, 3]
                ax_cbar.axis('off')

            ax_size = ax_merge(_ax)[:6, 3]
            ax_size.axis('off')
            self._plot_size_legend(ax_size)

            if self.figure_title is not None:
                ax.set_title(self.figure_title)

    def _plot_ax(self, x, y, values, **kwargs):
        """Plot wit existing Axes"""
        if self.with_color:
            im = self.ax.scatter(x.ravel(), y.ravel(),
                                 s=self.size,
                                 c=values,
                                 cmap=self.cmap,
                                 clip_on=False,
                                 norm=self.norm,
                                 edgecolor='gray',
                                 **kwargs)

            cbar = self.ax.figure.colorbar(im)
            cbar.ax.set_ylabel(self.colorbar_title)

        else:
            self.ax.scatter(x.ravel(), y.ravel(),
                            s=self.size,
                            c='k',
                            clip_on=False,
                            **kwargs)

        if self.figure_title is not None:
            self.ax.set_title(self.figure_title)

        self._plot_size_legend(self.ax)

    def _plot_size_legend(self, size_ax: Axes):
        values = np.round(
            np.linspace(self.norm.vmin, self.norm.vmax, num=self.size_legend_num), 2
        )

        if self.size_as_int:
            values = np.round(values).astype(np.int_)

        for v in values:
            if self.scale == 'radius':
                size = (v ** 2) * self.scaling_factor
            elif self.scale == 'area':
                size = v * self.scaling_factor
            else:
                raise ValueError('')

            size_ax.scatter([], [], s=size, c='k', label=str(v))

        handle, label = size_ax.get_legend_handles_labels()

        size_ax.legend(handle, label, labelspacing=1.2, title=self.size_title, borderpad=1,
                       frameon=True, framealpha=0.6, edgecolor="k", facecolor="w", loc='right')

    def _plot_colorbar(self, cbar_ax: Axes,
                       normalize: mcolors.Normalize) -> None:
        """Plots a horizontal colorbar with normalize values"""
        cmap = plt.get_cmap(self.cmap)

        mappable = ScalarMappable(norm=normalize, cmap=cmap)

        matplotlib.colorbar.Colorbar(
            cbar_ax,
            mappable=mappable,
            orientation="horizontal"
        )

        cbar_ax.set_title(self.colorbar_title, fontsize="small")

        cbar_ax.xaxis.set_tick_params(labelsize="small")
