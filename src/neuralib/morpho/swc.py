"""
Swc morphology plot
====================


**Example for CLI usage**

.. code-block:: console

    python -m neuralib.morpho.swc -h


**3D view with radius**

- Press ``Shift+s`` for vedo interactive saving

.. code-block:: console

    python -m neuralib.morpho.swc <SWC_FILE> --radius

**2D view with radius**

.. code-block:: console

    python -m neuralib.morpho.swc <SWC_FILE> --radius --2d


"""

from pathlib import Path
from typing import NamedTuple, Iterator

import matplotlib.pyplot as plt
import numpy as np
import vedo
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from neuralib.argp import AbstractParser, argument
from neuralib.typing import PathLike
from typing_extensions import Self, overload

__all__ = [
    'SwcNode',
    'SwcFile',
    'plot_swc'
]

Identifier = int
IdentifierName = str

IDENTIFIER_DICT: dict[Identifier, IdentifierName] = {
    0: 'undefined',
    1: 'soma',
    2: 'axon',
    3: 'basal',
    4: 'apical',
    5: 'custom'
}


class SwcNode(NamedTuple):
    n: int
    """node number"""

    identifier: Identifier
    """See IDENTIFIER_DICT"""

    x: float
    """position x"""
    y: float
    """position y"""
    z: float
    """position z"""
    r: float
    """radius"""
    parent: int
    """parent connectivity"""

    @property
    def identifier_name(self) -> IdentifierName:
        return IDENTIFIER_DICT.get(self.identifier, 'custom')

    @property
    def point(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def is_undefined(self) -> bool:
        return self.identifier == 0

    @property
    def is_soma(self) -> bool:
        return self.identifier == 1

    @property
    def is_axon(self) -> bool:
        return self.identifier == 2

    @property
    def is_basal_dendrite(self) -> bool:
        return self.identifier == 3

    @property
    def is_apical_dendrite(self) -> bool:
        return self.identifier == 4

    @property
    def is_dendrite(self) -> bool:
        return self.is_basal_dendrite or self.is_apical_dendrite

    @property
    def is_custom(self) -> bool:
        return self.identifier >= 5


class SwcFile:
    node: list[SwcNode]

    def __init__(self, node: list[SwcNode]):
        self.node = node

    @classmethod
    def load(cls, file: PathLike) -> Self:
        """
        :param file: swc filepath
        :return: ``SwcFile``
        """
        node = []
        with Path(file).open('r', encoding='Big5') as f:
            for line in f:
                line = line.strip()

                if len(line) == 0 or line.startswith('#'):
                    continue

                part = line.split()
                n = int(part[0])
                i = int(part[1])
                x = float(part[2])
                y = float(part[3])
                z = float(part[4])
                r = float(part[5])
                p = int(part[6])

                node.append(SwcNode(n, i, x, y, z, r, p))

        return cls(node)

    def __str__(self):
        line = [str(node) for node in self.node]
        return '\n'.join(line)

    @overload
    def __getitem__(self, item: int) -> SwcNode:
        pass

    @overload
    def __getitem__(self, item: IdentifierName) -> Self:
        pass

    def __getitem__(self, item: int | str) -> SwcNode | Self:
        if isinstance(item, int):
            try:
                ret = self.node[item - 1]  # to index
            except IndexError:
                ret = None

            if ret is not None and ret.n == item:
                return ret

            raise KeyError(f'item not found: {item}, might be loss parent connection')

        elif isinstance(item, str):
            if item == 'soma':
                node = [n for n in self.foreach_node() if n.is_soma]
            elif item == 'axon':
                node = [n for n in self.foreach_node() if n.is_axon]
            elif item == 'dendrite':
                node = [n for n in self.foreach_node() if n.is_dendrite]
            elif item == 'basal':
                node = [n for n in self.foreach_node() if n.is_basal_dendrite]
            elif item == 'apical':
                node = [n for n in self.foreach_node() if n.is_apical_dendrite]
            elif item == 'dendrite':
                node = [n for n in self.foreach_node() if n.is_dendrite]
            elif item == 'custom':
                node = [n for n in self.foreach_node() if n.is_custom]
            elif item == 'undefined':
                node = [n for n in self.foreach_node() if n.is_undefined]
            else:
                raise ValueError('')

            return SwcFile(node)

        else:
            raise TypeError(f'item must be int or str: {type(item)}')

    @property
    def points(self) -> np.ndarray:
        return np.array([[n.x, n.y, n.z] for n in self.foreach_node()])

    @property
    def radii(self) -> np.ndarray:
        return np.array([n.r for n in self.foreach_node()])

    @property
    def parents(self) -> np.ndarray:
        return np.array([n.parent for n in self.foreach_node()])

    @property
    def unique_identifier(self) -> list[IdentifierName]:
        idfs = np.unique([n.identifier for n in self.foreach_node()])
        return [
            IDENTIFIER_DICT.get(idf, 'custom')
            for idf in idfs
        ]

    def foreach_identifier(self, as_dict: bool) -> list[Self] | dict[str, Self]:
        if as_dict:
            return {idf: self[idf] for idf in self.unique_identifier}
        else:
            return [self[idf] for idf in self.unique_identifier]

    def foreach_node(self) -> Iterator[SwcNode]:
        for node in self.node:
            yield node

    def foreach_line(self) -> Iterator[tuple[SwcNode, SwcNode]]:
        for node in self.node:
            if node.parent > 0:
                yield node, self[node.parent]


# ============== #
# Plot Functions #
# ============== #

Point3D = tuple[float, float, float]
Point2D = tuple[float, float]

DEFAULT_COLOR: dict[IdentifierName, str] = {
    'soma': 'b',
    'axon': 'r',
    'dendrite': 'k',
    'undefined': 'k',
    'custom': 'k'
}


def projection_2d(p: Point3D) -> Point2D:
    """Default projection function, remove z value.

    :param p: 3d points
    :return: 2d points
    """
    return p[0], p[1]


def smooth_line_radius(ax: Axes,
                       p1: Point2D,
                       p2: Point2D,
                       r1: float,
                       r2: float,
                       num: int = 2,
                       **kwargs):
    """

    :param ax: ``Axes``
    :param p1: Point 1
    :param p2: Point 2
    :param r1: Radius 1
    :param r2: Radius 2
    :param num: Number of segments
    :param kwargs: Additional arguments pass to ``plt.plot()``
    :return:
    """
    px = np.linspace(p1[0], p2[0], num + 1)
    py = np.linspace(p1[1], p2[1], num + 1)
    lw = np.linspace(r1, r2, num)
    for i in range(num):
        ax.plot(px[i:i + 2], py[i:i + 2], lw=lw[i], **kwargs)


def plot_swc(swc: SwcFile,
             radius: bool = True,
             color: dict[str, str] | None = None,
             as_2d: bool = False):
    """
    Plot swc file as 2d

    :param swc: ``SwcFile``
    :param radius: Plot with radius.
    :param color: Color dict. With {identifier name: color coded}
    :param as_2d:
    """
    if color is None:
        color = DEFAULT_COLOR

    if as_2d:
        _plot_swc_2d(swc, radius, color)
    else:
        _plot_swc_3d(swc, radius, color)


def _plot_swc_2d(swc, radius, color):
    fig, ax = plt.subplots()
    for n1, n2 in swc.foreach_line():
        c = color.get(n1.identifier_name, 'k')

        p1 = projection_2d((n1.x, n1.y, n1.z))
        p2 = projection_2d((n2.x, n2.y, n2.z))

        if radius:
            if n2.is_soma:
                ax.add_artist(Circle(p2, n2.r, color=color['soma']))
                if not n1.is_soma:
                    smooth_line_radius(ax, p1, p2, n1.r, n1.r, color=c, solid_capstyle='round')
            else:
                smooth_line_radius(ax, p1, p2, n1.r, n2.r, color=c, solid_capstyle='round')
        else:
            px = p1[0], p2[0]
            py = p1[1], p2[1]
            ax.plot(px, py, color=c, solid_capstyle='round')

    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def _plot_swc_3d(swc: SwcFile,
                 radius,
                 color,
                 spheres_size: float = 3,
                 lw: float = 5):
    plotter = vedo.Plotter()

    axons = []
    axons_line = []
    axons_radii = []

    dendrites = []
    dendrites_line = []
    dendrites_radii = []

    somata = []
    somata_line = []
    somata_radii = []

    other = []
    other_line = []
    other_radii = []

    for i, n in enumerate(swc.foreach_node()):
        if n.parent == -1:
            continue

        r = n.r * spheres_size if radius else 5

        if n.is_axon:
            axons.append([n.x, n.y, n.z])
            axons_line.append([n.parent - 1, i])  # Use parent-child connection for axons
            axons_radii.append(r)
        elif n.is_dendrite:
            dendrites.append([n.x, n.y, n.z])
            dendrites_line.append([n.parent - 1, i])
            dendrites_radii.append(r)
        elif n.is_soma:
            somata.append([n.x, n.y, n.z])
            somata_line.append([n.parent - 1, i])
            somata_radii.append(10)  # fix value

        elif n.is_undefined or n.is_custom:
            other.append([n.x, n.y, n.z])
            other_line.append([n.parent - 1, i])
            other_radii.append(r)

    #
    if 'soma' in swc.unique_identifier:
        soma_spheres = vedo.Spheres(somata, r=somata_radii, c=color['soma'])
        soma_lines = vedo.Lines(swc.points[somata_line], c=color['soma'], lw=lw)
        plotter += soma_spheres
        plotter += soma_lines

    if 'dendrite' in swc.unique_identifier:
        dendrite_spheres = vedo.Spheres(dendrites, r=dendrites_radii, c=color['dendrite'])
        dendrite_lines = vedo.Lines(swc.points[dendrites_line], c=color['dendrite'], lw=lw)
        plotter += dendrite_spheres
        plotter += dendrite_lines

    if 'axon' in swc.unique_identifier:
        axon_spheres = vedo.Spheres(axons, r=axons_radii, c=color['axon'])
        axon_lines = vedo.Lines(swc.points[axons_line], c=color['axon'], lw=lw)
        plotter += axon_spheres
        plotter += axon_lines

    other_spheres = vedo.Spheres(other, r=other_radii, c=color.get('custom', 'k'))
    other_lines = vedo.Lines(swc.points[other_line], c=color.get('custom', 'k'), lw=lw)
    plotter += other_spheres
    plotter += other_lines

    plotter.show()


# ======== #
# Plot CLI #
# ======== #

class SwcPlotOptions(AbstractParser):
    file: str = argument(
        metavar='FILE',
        help='filepath of the swc file'
    )

    radius: bool = argument(
        '-R', '--radius',
        help='Whether plot with radius'
    )

    as_2d: bool = argument(
        '--2d',
        help='Whether plot with 2d, otherwise, plot as 3d'
    )

    def run(self):
        swc = SwcFile.load(self.file)
        plot_swc(swc, radius=self.radius, as_2d=self.as_2d)


if __name__ == '__main__':
    SwcPlotOptions().main()
