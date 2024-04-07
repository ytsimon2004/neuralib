from __future__ import annotations

import polars as pl
from bg_atlasapi import BrainGlobeAtlas
from plotly import express as px

from neuralib.util.util_type import PathLike


def plot_sunburst_acronym(source: str = 'allen_mouse_10um',
                          check_latest: bool = True,
                          output: PathLike | None = None):
    """
    plot allen brain structure tree interactive plot

    :param source: allen source name
    :param check_latest: if check the brainglobe api latest version
    :param output: figure output path, otherwise, render interactively
    :return:
    """
    file = BrainGlobeAtlas(source, check_latest=check_latest).root_dir / 'structures.csv'
    df = pl.read_csv(file)

    name = df.select(pl.col('acronym').alias('name'), pl.col('id'), pl.col('parent_structure_id').cast(int))
    xx = name.join(name, left_on='parent_structure_id', right_on='id')
    yy = xx.select(pl.col('name'), pl.col('name_right').alias('parent'))

    data = dict(
        character=yy.get_column('name'),
        parent=yy.get_column('parent'),
    )

    #
    fig = px.sunburst(
        data,
        names='character',
        parents='parent',

    )
    if output is not None:
        fig.write_image(output)
    else:
        fig.show()


if __name__ == '__main__':
    plot_sunburst_acronym()
