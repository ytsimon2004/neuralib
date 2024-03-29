from __future__ import annotations

from typing import Optional, Any, Literal, NamedTuple

from matplotlib.axes import Axes


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

    def with_total(self, total: int | Literal['default']) -> 'VennHandler':
        """total set number. should include the non-classified population.
            If None, then use the sum of the given subsets"""
        if total == 'default':
            total = self.subset_a + self.subset_b + self.subset_overlap

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


def plot_venn(ax: Axes,
              vhandler: VennHandler,
              msg: Optional[list[Any]] = None,
              labels: tuple[str, str] = ('visual', 'place'),
              show_msg=True):
    """
    three classified population venn diagram, and calculate the chance level

    :param ax:
    :param vhandler: VennHandler
    :param msg: msg show in title
    :param labels: label inside the venn
    :param show_msg:
    :return:
    """
    from matplotlib_venn import venn2

    venn2(ax=ax, subsets=vhandler.get_pure_number(), set_labels=labels, set_colors=('pink', 'palegreen'))

    title = []
    if show_msg:
        vfrac, pfrac, ofrac = vhandler.get_pure_fraction()
        title.append(f'chance: {round(vhandler.chance_level, 2)}%')
        title.append(f'fraction: {round(vfrac, 2)}%, {round(ofrac, 2)}%, {round(pfrac, 2)}%')
        title.append(f'total: {vhandler.total_set}')

    if msg is not None:
        title.extend(map(str, msg))

    title = '\n'.join(title)

    ax.set_title(title)

    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
