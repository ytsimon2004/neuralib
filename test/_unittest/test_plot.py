import unittest
from typing import Callable
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt

from neuralib.plot import plot_figure, dotplot, VennDiagram
from neuralib.plot.tools import AxesExtendHelper


# ================ #
# AxesExtendHelper #
# ================ #

def test_axes_extend_helper_x():
    x = np.random.sample(10)
    y = np.random.sample(10)
    with plot_figure(None) as ax:
        ax.plot(x, y, 'k.')

        helper = AxesExtendHelper(ax, mode='x')
        helper.xhist(x, bins=10)


def test_axes_extend_helper_y():
    x = np.random.sample(10)
    y = np.random.sample(10)
    with plot_figure(None) as ax:
        ax.plot(x, y, 'k.')

        helper = AxesExtendHelper(ax, mode='y')
        helper.yhist(y, bins=10)


def test_axes_extend_helper_hist():
    x = np.random.sample(10)
    y = np.random.sample(10)
    with plot_figure(None) as ax:
        ax.plot(x, y, 'k.')

        helper = AxesExtendHelper(ax)
        helper.xhist(x, bins=10)
        helper.yhist(y, bins=10)


def test_axes_extend_helper_bar():
    img = np.random.sample((10, 10))
    with plot_figure(None) as ax:
        ax.imshow(img)

        helper = AxesExtendHelper(ax)
        x = y = np.arange(10)
        helper.xbar(x, np.mean(img, axis=0), align='center')
        helper.ybar(y, np.mean(img, axis=1), align='center')


# ================= #
# Other Plots Usage #
# ================= #

def test_dotplot():
    xlabel = ['animal_A', 'animal_B', 'animal_C']
    ylabel = ['VISam', 'VISp', 'VISpm', 'VISpor', 'VISl', 'VISal', 'VISli', 'VISpl']
    nx = len(xlabel)
    ny = len(ylabel)
    values = np.random.sample((nx, ny)) * 100

    dotplot(xlabel, ylabel, values, max_marker_size=700, with_color=True, scale='area', figure_title='example dotplot')


def test_dotplot_ax():
    xlabel = ['animal_A', 'animal_B', 'animal_C']
    ylabel = ['VISam', 'VISp', 'VISpm', 'VISpor', 'VISl', 'VISal', 'VISli', 'VISpl']
    nx = len(xlabel)
    ny = len(ylabel)
    values = np.random.sample((nx, ny)) * 100

    with plot_figure(None, 2, 1) as _ax:
        ax = _ax[0]
        dotplot(xlabel, ylabel, values, max_marker_size=700, with_color=True, scale='area', ax=ax)
        ax.set_title('test_ax_plot')

        ax = _ax[1]
        ax.plot([1, 2, 3])


def test_venn2():
    subsets = {'setA': 10, 'setB': 20}
    vd = VennDiagram(subsets, colors=('pink', 'palegreen'))
    vd.add_intersection('setA & setB', 5)
    vd.add_total(100)
    vd.plot()
    vd.show()


def test_venn3():
    subsets = {'setA': 20, 'setB': 100, 'setC': 50}
    vd = VennDiagram(subsets)
    vd.add_intersection('setA & setB', 10)
    vd.add_intersection('setB & setC', 10)
    vd.add_intersection('setA & setC', 10)
    vd.add_intersection('setA & setB & setC', 2)
    vd.add_total(200)
    vd.plot()
    vd.show()


class TestPlotting(unittest.TestCase):

    @patch('matplotlib.pyplot.show')
    def plt_close(self, f: Callable, mock_show, *args, **kwargs):
        try:
            f(*args, **kwargs)
            plt.clf()
            plt.close('all')
        except Exception as e:
            self.fail(f'Plotting function raised an exception: {e}')

    def test_plotting_func_runs(self):
        self.plt_close(test_axes_extend_helper_x)
        self.plt_close(test_axes_extend_helper_y)
        self.plt_close(test_axes_extend_helper_hist)
        self.plt_close(test_axes_extend_helper_bar)
        self.plt_close(test_dotplot)
        self.plt_close(test_dotplot_ax)
        self.plt_close(test_venn2)
        self.plt_close(test_venn3)


if __name__ == '__main__':
    unittest.main()
