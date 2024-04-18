from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

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
                win_backend: MPL_BACKEND_TYPE = 'QtCairo',
                dpi: int | None = None,
                use_default_style: bool = True,
                **kwargs) -> Axes:
    """

    **Example**

    >>> fig_output = Path('output.png')
    >>> with plot_figure(fig_output) as ax:
    ...     ax.plot([0, 10], [0, 10])

    generate output.png

    :param output: output figure path
    :param args: plt.subplots(*args)
    :param set_square: sqaure plot
    :param set_equal_scale: equal for xy scaling. i.e., annotomical plot
    :param win_backend: handle backend problem in win10 pdf output.
        if high resolution image (WXAgg), otherwise keep normal pdf output (WXCario)
    :param dpi
    :param use_default_style:
    :param kwargs: plt.subplots(**kwargs)
    :return:
    """
    _os_handler(win_backend=win_backend)
    _mplrc_set()

    fig, ax = plt.subplots(*args, **kwargs)

    #
    if use_default_style:
        if isinstance(ax, np.ndarray):
            for _ax in ax.ravel():
                ax_set_default_style(_ax, set_square=set_square, set_equal_scale=set_equal_scale)
        else:
            ax_set_default_style(ax, set_square=set_square, set_equal_scale=set_equal_scale)
    #
    try:
        yield ax
    except:
        raise
    else:
        if output is None:
            plt.tight_layout()
            plt.show()
        else:
            while True:
                try:
                    # UserWarning: This figure includes Axes that are not compatible with tight_layout,
                    # so results might be incorrect
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        plt.tight_layout()
                except:
                    pass

                try:
                    plt.savefig(output, dpi=dpi if dpi is not None else None)
                    break
                except OSError as e:
                    print(e)
                    input('press to continue')

    finally:
        plt.clf()
        plt.close('all')


def _os_handler(win_backend: MPL_BACKEND_TYPE = 'WXCairo'):
    import platform
    import matplotlib as mpl

    if platform.system() == 'Windows':
        mpl.use(win_backend)


def _mplrc_set():  # TODO read from rc file with plt.rc_context()
    plt.rcParams['font.sans-serif'] = "Arial"


def ax_set_default_style(ax: Axes,  # TODO read from rc file with plt.rc_context()
                         set_square=False,
                         set_equal_scale=False):
    if 'polar' in ax.spines.keys():
        pass
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_tick_params(width=1)
        ax.yaxis.set_tick_params(width=1)
        for axis in ['bottom', 'left']:
            ax.spines[axis].set_linewidth(1)

    if set_square:
        ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    if set_equal_scale:
        ax.set_aspect('equal')


# ========= #
# AxesMerge #
# ========= #


class AxesMergeHelper:
    def __init__(self, ax: np.ndarray):
        self.__fig = ax.ravel()[0].figure
        self.__gs = ax.ravel()[0].get_gridspec()
        self.__ax = ax

    def __getitem__(self, item) -> Axes:
        for ax in self.__ax[item].ravel():
            ax.set_visible(False)
            # ax.remove()

        ret = self.__fig.add_subplot(self.__gs[item])
        ax_set_default_style(ret)
        return ret


def ax_merge(ax: np.ndarray) -> AxesMergeHelper:  # TODO give doc with examples
    return AxesMergeHelper(ax)
