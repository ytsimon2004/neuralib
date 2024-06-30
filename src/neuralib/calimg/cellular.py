from __future__ import annotations

import typing
from typing import Literal

import attrs
import numpy as np
from typing_extensions import Self

from neuralib.calimg import ObjectiveFov

__all__ = ['CellularCoordinates']


@typing.final
@attrs.define
class CellularCoordinates:
    """Container with coordinates information (in mm) for each ROIs"""

    neuron_idx: np.ndarray
    """neuron index"""
    ap: np.ndarray
    """anterior posterior coordinates (default in mm)"""
    ml: np.ndarray
    """medial lateral coordinates (default in mm)"""

    #
    plane_index: int | np.ndarray | None = attrs.field(default=None, kw_only=True)
    """optic plane index. i.e., used for depth analysis"""

    value: np.ndarray | None = attrs.field(default=None, kw_only=True)
    """metric (i.e., used in topographical analysis)"""

    unit: Literal['mm', 'um'] = attrs.field(default='mm', kw_only=True, validator=attrs.validators.in_(('mm', 'um')))

    def to_um(self) -> Self:
        """unit from mm to um"""
        if self.unit == 'um':
            raise RuntimeError('unit already in um')

        return attrs.evolve(
            self,
            ap=self.ap * 1000,
            ml=self.ml * 1000,
            unit='um'
        )

    def with_value(self, value: np.ndarray) -> Self:
        """
        assign ``value`` for ``CellularCoordinates``

        :param value: value array for the corresponding coordinates
        :return:
        """

        assert len(value) == len(self.ap) == len(self.ml)
        return attrs.evolve(self, value=value)

    def with_selection(self, mask: np.ndarray) -> Self:
        """masking for cell selection

        :param mask: numpy bool array
        """
        return attrs.evolve(
            self,
            neuron_idx=self.neuron_idx[mask],
            ap=self.ap[mask],
            ml=self.ml[mask],
            value=None if self.value is None else self.value[mask]
        )

    def in_relative_bregma(self, fov: ObjectiveFov) -> Self:
        """TODO check
        register cellular coordinates to IBL coordinates space

        :param fov: :class:`~neuralib.calimg.fov.ObjectiveFov`
        """
        if fov.rotation_angle_ap != 0:
            raise NotImplementedError('')

        fov = fov.to_um()

        zelf = self.to_um() if self.unit != 'um' else self

        if fov.rotation_angle_ml != 0:
            rx, ry = self.ml_rotate(zelf.ml, zelf.ap, fov.rotation_angle_ml)
            return attrs.evolve(
                self,
                ap=fov.am[1] - ry,
                ml=fov.am[0] - rx
            )
        else:
            return attrs.evolve(
                self,
                ap=fov.am[1] - zelf.ap,
                ml=fov.am[0] - zelf.ml
            )

    # ========================= #
    # Handle Objective rotation #
    # ========================= #

    # TODO need further testing. the drifted in this method increased across depth
    @staticmethod
    def ml_rotate(x: np.ndarray, y: np.ndarray, deg: float) -> tuple[np.ndarray, np.ndarray]:
        """
        calculate the values for x axis rotation

        :param x: coordinate x (N,)
        :param y: coordinate Y (N,)
        :param deg: rotation degree
        :return: value to be subtracted. dx (N,) and dy (N,)
        """
        angle = np.radians(deg)
        dx = x * np.cos(angle) - y * np.sin(angle)
        dy = x * np.sin(angle) + y * np.cos(angle)

        return dx, dy

    @staticmethod
    def ap_rotate():
        # TODO
        pass
