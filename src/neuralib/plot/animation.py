from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from neuralib.util.util_type import PathLike

__all__ = ['plot_scatter_animation']


def plot_scatter_animation(x: np.ndarray,
                           y: np.ndarray,
                           t: np.ndarray | None = None, *,
                           step: int | None = None,
                           size: int = 10,
                           output: PathLike | None = None,
                           **kwargs) -> None:
    """
    Plot xy scatter animation with given time points

    :param x: x loc (T, )
    :param y: y loc (T, )
    :param t: time array in sec (T, )
    :param size: size of the scatter
    :param step: step run per datapoint
    :param output: output for animation. i.e., *.gif
    :param kwargs: FuncAnimation(**kwargs)
    :return:
    """
    from matplotlib.animation import FuncAnimation

    fig, _ = plt.subplots()

    def foreach_run(frame: int):
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))

        if step is not None:
            frame *= step

        ax.text(0.02, 0.95, f'Frames = {frame}', transform=ax.transAxes)
        if t is not None:
            ax.text(0.02, 0.85, f'Time = {t[frame]:.2f}', transform=ax.transAxes)

        ax.scatter(x[frame], y[frame], s=size)

    ani = FuncAnimation(fig, foreach_run, frames=len(x), **kwargs)

    try:
        if output is not None:
            ani.save(output)
        else:
            plt.show()

    finally:
        plt.clf()
        plt.close('all')
