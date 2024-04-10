from __future__ import annotations

import colorsys
from typing import TypedDict

import numpy as np

from neuralib.plot import plot_figure
from neuralib.util.util_type import PathLike

__all__ = [
    'plot_task_gantt',
    'draw_circular_annotation',
    'generate_dots',
    'draw_random_dots'
]


class GanttDict(TypedDict, total=False):
    task: str
    start: str
    finish: str
    resource: str


def plot_task_gantt(jobs: list[GanttDict]):
    """
    plot gantt
    * Example

    >>> x = [GanttDict(task='task1', start='2023-09-01', finish='2023-11-30'),
    ...      GanttDict(task='task2', start='2024-01-01', finish='2024-09-30')]
    >>> plot_task_gantt(x)

    :param jobs:
    :return:
    """
    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame(jobs)
    if 'resource' not in df.columns:
        color = 'task'
    else:
        color = 'resource'
    fig = px.timeline(df, x_start='start', x_end='finish', y='task', color=color)
    fig.update_layout(
        width=1200,
        height=400
    )
    fig.show()


# ================= #

def draw_circular_annotation(major_radius: float = 3,
                             minor_radius: float = 2,
                             width: float = 10,
                             output: PathLike | None = None):
    """plot circular color annotation. i.e., used in retinotopic mapping illustration"""
    center_x, center_y = 0, 0
    n_points = 1000
    theta = np.linspace(0, 2 * np.pi, n_points)

    x = center_x + major_radius * np.cos(theta)
    y = center_y + minor_radius * np.sin(theta)

    hues = np.linspace(0, 1, n_points)
    colors = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in hues]

    with plot_figure(output) as ax:
        for p in range(n_points - 1):
            ax.plot(x[p:p + 2], y[p:p + 2], color=colors[p], linewidth=width)

        ax.set_aspect('equal', adjustable='datalim')

        ax.set_xlim(-major_radius - 1, major_radius + 1)
        ax.set_ylim(-minor_radius - 1, minor_radius + 1)
        ax.axis('off')


# ================= #

def generate_dots(n_dots: int = 20,
                  min_distance: float = 0.1,
                  border_distance: float = 0.1):
    """

    :param n_dots:
    :param min_distance: minimum distance to avoid overlap
    :param border_distance:  minimum distance from the border
    :return:
    """
    ret = []
    existing_dots = []
    while len(ret) < n_dots:
        x = np.random.uniform(border_distance, 1 - border_distance)
        y = np.random.uniform(border_distance, 1 - border_distance)

        # Check for overlap with existing coordinates
        overlap = False
        for coord in existing_dots:
            if np.sqrt((x - coord[0]) ** 2 + (y - coord[1]) ** 2) < min_distance:
                overlap = True
                break

        if not overlap:
            ret.append((x, y))
            existing_dots.append((x, y))

    return ret


def draw_random_dots(n_dots: int = 20,
                     min_distance: float = 0.1,
                     border_distance: float = 0.01,
                     output: PathLike | None = None,
                     pixels: tuple[int, int] = (1000, 1000)):
    dots = generate_dots(n_dots, min_distance, border_distance)

    dpi = 500
    with plot_figure(output,
                     figsize=(pixels[0] / dpi, pixels[1] / dpi), dpi=dpi) as ax:
        for d in dots:
            ax.scatter(d[0], d[1], color='black', s=20)

        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_facecolor('white')
        ax.axis('off')
        ax.set_aspect('equal', adjustable='box')
