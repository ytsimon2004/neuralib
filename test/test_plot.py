import numpy as np
from matplotlib import pyplot as plt

from neuralib.plot import plot_figure, dotplot
from neuralib.plot._test import scattermap
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


def test_seaborn_dot():
    values = np.array([
        [35.33418476, 22.90336313, 16.85823755, 13.40996169, 3.53341848, 5.57684121, 1.8305662, 0.55342699],
        [25.14484357, 27.5202781, 26.36152955, 7.24217845, 5.79374276, 1.76709154, 3.47624565, 2.69409038],
        [43.33418476, 5.90336313, 12.85823755, 23.40996169, 1.53341848, 7.57684121, 2.8305662, 0.55342699],
    ])

    ax = scattermap(values, marker_size=values * 10, square=False, cmap="Reds",
                    cbar_kws={"label": "Color data"},
                    yticklabels=['animal_A', 'animal_B', 'animal_C'],
                    xticklabels=['VISam', 'VISp', 'VISpm', 'VISpor', 'VISl', 'VISal', 'VISli', 'VISpl'])

    mk_size = np.max(values) * 10

    ax.scatter(-1, -1, label=f"{np.amax(values):0.1f}", marker="o", c="r", s=mk_size)
    ax.scatter(-1, -1, label=f"{np.mean(values):0.1f}", marker="o", c="r", s=mk_size * 0.5)
    ax.scatter(-1, -1, label=f"{np.amin(values[np.nonzero(values)]):0.1f}", marker="o", c="r", s=mk_size * 0.1)
    ax.legend(loc="upper left", bbox_to_anchor=(0.97, -0.05))

    ax.text(10.65, 11, "Size data", rotation=90, fontsize="medium")
    plt.show()


if __name__ == '__main__':
    test_dotplot_ax()
