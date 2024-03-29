from __future__ import annotations

from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.transforms import Transform

from neuralib.util.util_type import PathLike

__all__ = [
    'AnchoredScaleBar',
    'AxesExtendHelper',
    'insert_latex_equation'
]


class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(
            self,
            transform: Transform,
            sizex: float = 0,
            sizey: float = 0,
            labelx: str | None = None,
            labely: str | None = None,
            loc: int | str = 4,
            pad: float = 0.1,
            borderpad=0.1,
            sep: int = 2,
            color: str = "black",
            lw: float = 1.5,
            color_txt: str = "black",
            prop=None,
            **kwargs
    ):
        """
        Set of scale bars that match the size of the ticks of the plot.

        Draws a horizontal and/or vertical bar with the size in data coordinates
        of the give axes. A label will be drawn underneath (center-aligned).

        :param transform: Matplotlib Transform
            The coordinate frame (typically axes.transData)
        :param sizex: Width of x bar, in data units. 0 to omit.
        :param sizey: Width of y bar, in data units. 0 to omit.
        :param labelx: Labels for x bars; None to omit
        :param labely: Labels for y bars; None to omit
        :param loc:  Position in containing axes.
        :param pad: Padding, in fraction of the legend font size (or prop).
        :param borderpad:
        :param sep: Separation between labels and bars in points.
        :param color: Bars color
        :param lw:  Bars width
        :param color_txt: color
        :param prop:  Font property.
        :param kwargs: additional arguments passed to base class constructor
        """

        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import (
            AuxTransformBox,
            VPacker,
            HPacker,
            TextArea,
        )

        bars = AuxTransformBox(transform)

        if sizex:
            bars.add_artist(Rectangle((0, 0), sizex, 0, fc="none", color=color, lw=lw))
        if sizey:
            bars.add_artist(Rectangle((0, 0), 0, sizey, fc="none", color=color, lw=lw))

        if sizex and labelx:
            bars = VPacker(children=[bars, TextArea(labelx, minimumdescent=False, textprops=dict(color=color_txt))],
                           align="center",
                           pad=0,
                           sep=sep)
        if sizey and labely:
            bars = HPacker(children=[TextArea(labely, textprops=dict(color=color_txt)), bars],
                           align="center",
                           pad=0,
                           sep=sep)

        AnchoredOffsetbox.__init__(
            self,
            loc,
            pad=pad,
            borderpad=borderpad,
            child=bars,
            prop=prop,
            frameon=False,
            **kwargs
        )


# ========================= #

class AxesExtendHelper:
    ax_x: Axes | None
    ax_y: Axes | None

    def __init__(self,
                 ax: Axes,
                 mode: Literal['both', 'x', 'y'] = 'both'):
        ax.set(aspect=1)
        if mode in ('both', 'x'):
            self.ax_x = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
        else:
            self.ax_x = None

        if mode in ('both', 'y'):
            self.ax_y = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
        else:
            self.ax_y = None

        self.ax = ax

    def xhist(self, values, bins, **kwargs):
        if self.ax_x is not None:
            self.ax_x.hist(values, bins, **kwargs)

    def yhist(self, values, bins, **kwargs):
        if self.ax_y is not None:
            self.ax_y.hist(values, bins, orientation='horizontal', **kwargs)

    def xbar(self, x, height, width: float | np.ndarray = None, **kwargs):

        default_kw = {
            'color': 'grey',
            'edgecolor': 'black',
            'align': 'edge'
        }

        kw = default_kw | kwargs

        if width is None:
            width = np.diff(x)[0]

        if self.ax_x is not None:
            self.ax_x.bar(x, height, width=width, **kw)

            self.ax_x.tick_params(axis="x", labelbottom=False)

    def ybar(self, y, width, height: float | np.ndarray = None, **kwargs):
        default_kw = {
            'color': 'grey',
            'edgecolor': 'black',
            'align': 'edge'
        }

        kw = default_kw | kwargs

        if height is None:
            height = np.diff(y)[0]

        if self.ax_y is not None:
            self.ax_y.barh(y, width, height=height, **kw)

            self.ax_y.tick_params(axis="y", labelleft=False)


# ========================= #

def insert_latex_equation(ax: Axes,
                          tex: str,
                          output: PathLike | None = None):
    """Plot figure with latex expression, and save as image"""
    plt.rcParams['text.usetex'] = True

    ax.set_title(tex)
    if output is not None:
        from matplotlib.mathtext import math_to_image
        math_to_image(tex, output, dpi=300)
