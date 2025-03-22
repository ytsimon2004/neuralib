
import platform
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, ContextManager

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import NDArray

__all__ = ['plot_figure',
           'ax_set_default_style',
           'ax_merge']

MPL_BACKEND_TYPE = Literal[
    'GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg',
    'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo',
    'TkAgg', 'TkCairo',
    'WebAgg',
    'WX', 'WXAgg', 'WXCairo'
]


@contextmanager
def plot_figure(output: Path | None,
                *args,
                set_square: bool = False,
                set_equal_scale: bool = False,
                win_backend: MPL_BACKEND_TYPE | None = 'QtCairo',
                dpi: int | None = None,
                default_style: bool = True,
                tight_layout: bool = True,
                font_sans_serif: str | None = 'Arial',
                **kwargs) -> ContextManager[Axes] | ContextManager[NDArray[Axes]]:
    """
    Context manager for creating and saving a matplotlib figure

    **Example**

    >>> fig_output = Path('output.png')
    >>> with plot_figure(fig_output) as ax:
    ...     ax.plot([0, 10], [0, 10])

    generate output.png

    :param output: Path to save the output figure. If None, the figure will be shown.
    :param args: Arguments for ``plt.subplots()``
    :param set_square: If True, set the plot to be square
    :param set_equal_scale: If True, set equal scaling for x and y axes
    :param win_backend: Backend to handle backend issues in Windows
        If high resolution image (WXAgg), otherwise keep normal pdf output (WXCario)
    :param dpi: DPI for saving the figure
    :param default_style: If True, apply default style to the axes
    :param tight_layout: If True, apply tight layout to the figure
    :param font_sans_serif: Font style. If None, then use default from mplrc
    :param kwargs: Additional keyword arguments for ``plt.subplots()``
    :return: Single or array of matplotlib Axes object
    """
    if win_backend is not None and platform.system() == 'Windows':
        matplotlib.use(win_backend)

    if font_sans_serif is not None:
        plt.rcParams['font.sans-serif'] = font_sans_serif

    fig, ax = plt.subplots(*args, **kwargs)

    if default_style:
        if isinstance(ax, np.ndarray):
            for _ax in ax.ravel():
                ax_set_default_style(_ax)
        else:
            ax_set_default_style(ax)

    #
    try:
        yield ax
    except Exception as e:
        raise RuntimeError(f'An error occurred while plotting {e}')
    else:

        if set_square:
            if isinstance(ax, np.ndarray):
                for _ax in ax.ravel():
                    _set_square(_ax)
            else:
                _set_square(ax)

        if set_equal_scale:
            if isinstance(ax, np.ndarray):
                for _ax in ax.ravel():
                    _ax.set_aspect('equal')
            else:
                ax.set_aspect('equal')

        # io
        if output is None:
            if tight_layout:
                plt.tight_layout()
            plt.show()

        else:
            while True:
                if tight_layout:
                    plt.tight_layout()

                # for batch calling pulse
                try:
                    plt.savefig(output, dpi=dpi)
                    break
                except OSError as e:
                    input(f'{repr(e)}! press to continue')

    finally:
        plt.clf()
        plt.close('all')


def _set_square(ax: Axes) -> None:
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')


def ax_set_default_style(ax: Axes):
    """
    Default `Axes` style

    :param ax: ``Axes``
    """
    if 'polar' in ax.spines.keys():
        pass
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_tick_params(width=1)
        ax.yaxis.set_tick_params(width=1)
        for axis in ['bottom', 'left']:
            ax.spines[axis].set_linewidth(1)


# ========= #
# AxesMerge #
# ========= #


class AxesMergeHelper:
    def __init__(self, ax: np.ndarray):
        """
        :param ax: `Array[Axes, G]`
        """
        self.__fig = ax.ravel()[0].figure
        self.__gs = ax.ravel()[0].get_gridspec()
        self.__ax = ax

    def __getitem__(self, item) -> Axes:
        for ax in self.__ax[item].ravel():
            ax.set_visible(False)

        ret = self.__fig.add_subplot(self.__gs[item])
        ax_set_default_style(ret)
        return ret


def ax_merge(ax: np.ndarray) -> AxesMergeHelper:
    """
    Subplots ``Axes`` merge

    `Dimension parameters`:

        R = number of rows

        C = number of columns

        G = R * C


    **Example of 5x3 grid subplots**

    +-----------------+-----------------+-----------------+
    |    Subplot 1    |    Subplot 1    |    Subplot 4    |
    |    ax1 (3x2)    |    ax1 (3x2)    |    ax4 (5x1)    |
    +-----------------+-----------------+-----------------+
    |    Subplot 1    |    Subplot 1    |    Subplot 4    |
    |    ax1 (3x2)    |    ax1 (3x2)    |    ax4 (5x1)    |
    +-----------------+-----------------+-----------------+
    |    Subplot 1    |    Subplot 1    |    Subplot 4    |
    |    ax1 (3x2)    |    ax1 (3x2)    |    ax4 (5x1)    |
    +-----------------+-----------------+-----------------+
    |    Subplot 2    |    Subplot 3    |    Subplot 4    |
    |    ax2 (2x1)    |    ax3 (2x1)    |    ax4 (5x1)    |
    +-----------------+-----------------+-----------------+
    |    Subplot 2    |    Subplot 3    |    Subplot 4    |
    |    ax2 (2x1)    |    ax3 (2x1)    |    ax4 (5x1)    |
    +-----------------+-----------------+-----------------+


    >>> with plot_figure(None, 5, 3) as _ax:
    ...     ax1 = ax_merge(_ax)[:3, :2]
    ...     ax1.plot(np.arange(5), 'ro')
    ...     ax1.set_title('subplot 1')
    ...
    ...     ax2 = ax_merge(_ax)[3:, 1]
    ...     ax2.imshow(np.random.sample((10, 10)))
    ...     ax2.set_title('subplot 2')
    ...
    ...     ax3 = ax_merge(_ax)[3:, 2]
    ...     ax3.plot(np.arange(10), 'g*')
    ...     ax3.set_title('subplot 3')
    ...
    ...     D = np.random.normal((3, 5, 4), (0.75, 1.00, 0.75), (200, 3))
    ...     ax4 = ax_merge(_ax)[:, -1]
    ...     ax4.violinplot(D, [2, 4, 6], widths=2, showmeans=False, showmedians=False, showextrema=False)
    ...     ax4.set_title('subplot 4')


    :param ax: `Array[Axes, G]`
    :return: ``AxesMergeHelper``
    """
    return AxesMergeHelper(ax)
