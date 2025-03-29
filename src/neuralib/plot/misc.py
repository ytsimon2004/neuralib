from typing import TypedDict

import polars as pl

__all__ = ['plot_task_gantt']


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

    :param jobs: list of ``GanttDict``
    """
    import plotly.express as px

    df = pl.DataFrame(jobs)
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
