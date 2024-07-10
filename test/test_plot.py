import numpy as np

from neuralib.plot import plot_figure
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


if __name__ == '__main__':
    test_axes_extend_helper_y()
