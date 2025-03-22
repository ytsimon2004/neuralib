from typing import Optional, NamedTuple, ClassVar

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from neuralib.typing import PathLike

__all__ = ['VennHandler',
           'VennDiagram']


class VennHandler(NamedTuple):
    subset_a: int
    """whole number with condition a"""
    subset_b: int
    """whole number with condition b"""
    subset_overlap: int
    """whole number with condition a & b"""

    total_set: Optional[int] = None

    @property
    def chance_level(self) -> float:
        if self.total_set is None:
            raise ValueError('call with_total first')

        fa = self.subset_a / self.total_set
        fb = self.subset_b / self.total_set

        return fa * fb * 100

    def with_total(self, total: int) -> 'VennHandler':
        """total set number. should include the non-classified population"""
        return self._replace(total_set=total)

    def get_pure_number(self) -> tuple[int, ...]:
        a = self.subset_a - self.subset_overlap
        b = self.subset_b - self.subset_overlap
        return tuple([a, b, self.subset_overlap])

    def get_pure_fraction(self) -> tuple[float, ...]:
        a, b, o = self.get_pure_number()
        return tuple([a / self.total_set * 100,
                      b / self.total_set * 100,
                      o / self.total_set * 100])


class VennDiagram:
    DEFAULT_COLORS: ClassVar[tuple[str, ...]] = ('r', 'g', 'b')

    def __init__(self,
                 subsets: dict[str, int],
                 *,
                 colors: tuple[str, ...] | None = None,
                 ax: Axes | None = None,
                 **kwargs):
        """

        :param subsets: Dictionary of set label and its value
        :param colors: colors of each venn
        :param ax: ``Axes``
        :param kwargs: additional args passed to ``matplotlib_venn.venn2()`` or ``matplotlib_venn.venn3()``
        """
        self.subsets = subsets

        self.total: int | None = None
        self.contain_intersection: bool = False
        self._intersections: dict[str, int] = {}

        # fig
        if colors is not None and len(colors) != len(self):
            raise ValueError('length of colors need to be the same as length of subset label')
        self.colors = colors or VennDiagram.DEFAULT_COLORS
        self.ax = ax

        self.kwargs = kwargs

    def __len__(self):
        return len(self.subsets)

    @property
    def intersections(self) -> dict[str, int]:
        """intersection for sets"""
        return self._intersections

    @property
    def max_intersection_areas(self):
        """maximal number of intersection areas"""
        if len(self) == 2:
            return 1
        elif len(self) == 3:
            return 4
        else:
            raise NotImplementedError('')

    @property
    def labels(self) -> tuple[str, ...]:
        """set names"""
        return tuple(self.subsets.keys())

    @property
    def subsets_percentage(self) -> dict[str, float]:
        """percentage of each subset"""
        if self.total is None:
            raise RuntimeError('add total first')

        source = {
            k: round(v / self.total * 100, 2)
            for k, v in self.subsets.items()
        }

        inter = {
            k: round(v / self.total * 100, 2)
            for k, v in self.intersections.items()
        }

        return {**source, **inter}

    def add_total(self, value: int):
        """
        Add total value to the venn diagram
        :param value: value to be added
        """
        self.total = value

    def add_intersection(self, group: str, value: int):
        """
        Add intersection values using "&"

        :param group: i.e., `a & b`
        :param value: value of the intersection
        """

        src = [g.strip() for g in group.split('&')]

        for inter in list(self.intersections.keys()):
            if all(g in inter for g in src):
                raise ValueError('intersection value already existed')

        for g in src:
            if g not in self.subsets:
                raise ValueError(f"Set '{g}' does not exist in the subsets.")

        self._intersections['&'.join(src)] = value

    def get_chance_level(self, *label) -> float:
        if self.contain_intersection:
            raise RuntimeError('chance level should not contain intersection')

        if self.total is None:
            raise RuntimeError('add total first')

        x = 100
        for it in label:
            x *= self.subsets[it] / self.total

        return x

    def get_intersection(self, *label: str) -> int:
        """
        Get intersection value from labels

        :param label:i.e., `a & b`
        :return: intersection value
        """
        k = '&'.join([*label])
        return self.intersections.get(k, 0)

    def with_intersection(self):
        """Add intersection value into subsets"""
        if self.contain_intersection:
            raise RuntimeError('already contain intersection')

        ret = {}
        if len(self) == 2:
            inter = self.get_intersection(*self.labels)
            for k, v in self.subsets.items():
                ret[k] = v + inter
        elif len(self) == 3:
            ab = self.get_intersection(*self.labels[:2])
            ac = self.get_intersection(self.labels[0], self.labels[2])
            bc = self.get_intersection(*self.labels[1:])
            abc = self.get_intersection(*self.labels)
            for i, (k, v) in enumerate(self.subsets.items()):
                if i == 0:
                    ret[k] = v + ab + ac + abc
                elif i == 1:
                    ret[k] = v + ab + bc + abc
                elif i == 2:
                    ret[k] = v + ac + bc + abc
        else:
            raise NotImplementedError('')

        self.subsets = ret
        self.contain_intersection = True

    # ================ #
    # Plotting Methods #
    # ================ #

    def plot(self, add_title: bool = True):
        """
        Plot the venn diagram

        :param add_title: Add percentage information and total as title
        """
        if self.ax is None:
            self.ax = plt.gca()

        n_subsets = len(self)
        if n_subsets == 2:
            self._venn2()
        elif n_subsets == 3:
            self._venn3()

        if add_title and self.total is not None:
            self.ax.set_title(self.title)

        self.ax.set_axis_off()
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    @staticmethod
    def show():
        """Show figure"""
        plt.show()

    @staticmethod
    def savefig(output: PathLike):
        """
        Save figure

        :param output: fig output
        """
        plt.savefig(output)

    @property
    def title(self) -> str:
        """title of the plot"""
        ret = [
            f'percentage: {self.subsets_percentage}%',
            f'total: {self.total}'
        ]

        return '\n'.join(ret)

    # noinspection PyTypeChecker
    def _venn2(self):
        """subsets = (a, b, a&b)"""
        from matplotlib_venn import venn2
        subsets = list(self.subsets.values()) + [self.get_intersection(*self.labels)]

        venn2(subsets=tuple(subsets),
              set_labels=self.labels,
              set_colors=self.colors[:2],
              ax=self.ax,
              **self.kwargs)

    # noinspection PyTypeChecker
    def _venn3(self):
        """subsets = (a, b, a&b, c, a&c, b&c, a&b&c)"""
        from matplotlib_venn import venn3
        v = list(self.subsets.values())
        a = v[0]
        b = v[1]
        c = v[2]
        ab = self.get_intersection(*self.labels[:2])
        ac = self.get_intersection(self.labels[0], self.labels[2])
        bc = self.get_intersection(*self.labels[1:])
        abc = self.get_intersection(*self.labels)

        venn3(subsets=(a, b, ab, c, ac, bc, abc),
              set_labels=self.labels,
              set_colors=self.colors,
              ax=self.ax,
              **self.kwargs)
