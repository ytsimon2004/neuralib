from pathlib import Path

from neuralib.atlas.data import load_bg_structure_tree
from neuralib.typing import PathLike
from plotly import express as px

__all__ = ['plot_sunburst_acronym',
           'plot_structure_tree']


def plot_sunburst_acronym(output: PathLike | None = None):
    """
    Plot allen brain structure tree interactive plot

    :param output: figure output path, otherwise, render interactively
    :return:
    """
    data = load_bg_structure_tree(parse=True)
    data = dict(names=data['names'], parents=data['parents'])

    #
    fig = px.sunburst(
        data,
        names='names',
        parents='parents',

    )
    if output is not None:
        fig.write_image(output)
    else:
        fig.show()


def plot_structure_tree(output: PathLike | None = None) -> None:
    """Show tree for the brain structure

    :param output: output file txt. print if None
    """
    from anytree import Node, RenderTree

    df = load_bg_structure_tree(parse=True)

    nodes = {}
    for it in df.iter_rows(named=True):
        name = it['names']
        parent = it['parents']

        if name not in nodes:
            nodes[name] = Node(name)
        if parent not in nodes:
            nodes[parent] = Node(parent)

        nodes[name].parent = nodes[parent]

    #
    for pre, fill, node in RenderTree(nodes['root']):
        if output is not None:
            output = Path(output)

            if not output.exists():
                with Path(output).open('a') as file:
                    print("%s%s" % (pre, node.name), file=file)
            else:
                raise FileExistsError(f'{output}')
        else:
            print("%s%s" % (pre, node.name))

