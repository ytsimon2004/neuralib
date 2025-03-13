from typing import Iterable, Literal

import attrs
import numpy as np
from matplotlib.patches import Polygon
from typing_extensions import TypeAlias, Self

__all__ = ['ObjectiveFov']

Coordinates: TypeAlias = np.ndarray
"""Array[float, 2] in XY"""


@attrs.define
class ObjectiveFov:
    """
    Class for 2P Field Of View coordinates

    Use IBL coordinates space ::

        AP (+), anterior to the Bregma. (-) posterior to the Bregma

        ML (+), right hemisphere. (-) left hemisphere

    .. seealso::  `<https://int-brain-lab.github.io/iblenv/notebooks_external/atlas_working_with_ibllib_atlas.html#Coordinate-systems>`_


    """

    region_name: str
    """FOV name"""
    am: Coordinates
    """anteromedial (default in mm). `Array[float, 2]` in XY"""
    pm: Coordinates
    """posteromedial (default in mm). `Array[float, 2]` in XY"""
    al: Coordinates
    """anterolateral (default in mm). `Array[float, 2]` in XY"""
    pl: Coordinates
    """posterolateral (default in mm). `Array[float, 2]` in XY"""

    #
    rotation_angle_ml: float = attrs.field(default=0, kw_only=True)
    """objective ml direction rotation in deg"""
    rotation_angle_ap: float = attrs.field(default=0, kw_only=True)
    """objective ap direction rotation in deg"""

    #
    unit: Literal['mm', 'um'] = attrs.field(default='mm', kw_only=True, validator=attrs.validators.in_(('mm', 'um')))

    def __iter__(self) -> Iterable[Coordinates]:
        for c in (self.am, self.pm, self.pl, self.al):  # order for polygon
            yield c

    @property
    def ap_distance(self) -> float:
        """anterior to posterior distance of the FOV"""
        a = np.max([self.am[1], self.al[1]])
        p = np.max([self.pm[1], self.pl[1]])
        return np.abs(p - a)

    @property
    def ml_distance(self) -> float:
        """medial to lateral distance of the FOV"""
        m = np.max([self.am[0], self.pm[0]])
        l = np.max([self.al[0], self.pl[0]])  # noqa: E741
        return np.abs(l - m)

    def to_um(self) -> Self:
        """unit from mm to um"""
        if self.unit == 'um':
            raise RuntimeError('unit already in um')

        factor = 1000
        return attrs.evolve(
            self,
            am=self.am * factor,
            pm=self.pm * factor,
            al=self.al * factor,
            pl=self.pl * factor,
            unit='um'
        )

    def ap_invert(self) -> Self:
        """anterior posterior coordinates invert"""
        for c in self:
            c[1] = c[1] * -1
        return attrs.evolve(self, self.am, self.pm, self.al, self.pl)

    def ml_invert(self) -> Self:
        """medial lateral coordinates invert"""
        for c in self:
            c[0] = c[0] * -1
        return attrs.evolve(self, self.am, self.pm, self.al, self.pl)

    def invert_axis(self) -> Self:
        """invert both ap and ml axes"""
        self.ap_invert()
        self.ml_invert()
        return attrs.evolve(self)

    def to_polygon(self, **kwargs) -> Polygon:
        """coordinates to ``matplotlib.patches.Polygon``"""
        x = [c[0] for c in self]
        y = [c[1] for c in self]
        return Polygon(list(zip(x, y)), closed=True, edgecolor='r', facecolor='none', **kwargs)
