import unittest
from unittest.mock import patch

import numpy as np

from neuralib.plot import plot_figure, dotplot, VennDiagram
from neuralib.plot.tools import AxesExtendHelper


# ================ #
# AxesExtendHelper #
# ================ #

@patch("matplotlib.pyplot.show")
def test_axes_extend_helper_x(mock):
    x = np.random.sample(10)
    y = np.random.sample(10)
    with plot_figure(None) as ax:
        ax.plot(x, y, 'k.')

        helper = AxesExtendHelper(ax, mode='x')
        helper.xhist(x, bins=10)


@patch("matplotlib.pyplot.show")
def test_axes_extend_helper_y(mock):
    x = np.random.sample(10)
    y = np.random.sample(10)
    with plot_figure(None) as ax:
        ax.plot(x, y, 'k.')

        helper = AxesExtendHelper(ax, mode='y')
        helper.yhist(y, bins=10)


@patch("matplotlib.pyplot.show")
def test_axes_extend_helper_hist(mock):
    x = np.random.sample(10)
    y = np.random.sample(10)
    with plot_figure(None) as ax:
        ax.plot(x, y, 'k.')

        helper = AxesExtendHelper(ax)
        helper.xhist(x, bins=10)
        helper.yhist(y, bins=10)


@patch("matplotlib.pyplot.show")
def test_axes_extend_helper_bar(mock):
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
@patch("matplotlib.pyplot.show")
def test_dotplot(mock):
    xlabel = ['animal_A', 'animal_B', 'animal_C']
    ylabel = ['VISam', 'VISp', 'VISpm', 'VISpor', 'VISl', 'VISal', 'VISli', 'VISpl']
    nx = len(xlabel)
    ny = len(ylabel)
    values = np.random.sample((nx, ny)) * 100

    dotplot(xlabel, ylabel, values, max_marker_size=700, with_color=True, scale='area', figure_title='example dotplot')


@patch("matplotlib.pyplot.show")
def test_dotplot_ax(mock):
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


@patch("matplotlib.pyplot.show")
def test_venn2(mock):
    subsets = {'setA': 10, 'setB': 20}
    vd = VennDiagram(subsets, colors=('pink', 'palegreen'))
    vd.add_intersection('setA & setB', 5)
    vd.add_total(100)
    vd.plot()
    vd.show()


@patch("matplotlib.pyplot.show")
def test_venn3(mock):
    subsets = {'setA': 20, 'setB': 100, 'setC': 50}
    vd = VennDiagram(subsets)
    vd.add_intersection('setA & setB', 10)
    vd.add_intersection('setB & setC', 10)
    vd.add_intersection('setA & setC', 10)
    vd.add_intersection('setA & setB & setC', 2)
    vd.add_total(200)
    vd.plot()
    vd.show()


if __name__ == '__main__':
    unittest.main()
