from __future__ import annotations

from typing import Sequence, Literal

import attrs
import numpy as np
from matplotlib.patches import Polygon
from typing_extensions import Self

from neuralib.util.unstable import unstable

__all__ = ['get_field_of_view',
           'get_cellular_coordinate',
           'FieldOfView',
           'CellularCoordinates']

UNIT = Literal['mm', 'um']


def get_field_of_view(am: Sequence[float],
                      pm: Sequence[float],
                      pl: Sequence[float],
                      al: Sequence[float],
                      *,
                      rotation_ml: float = 0,
                      rotation_ap: float = 0,
                      unit: UNIT = 'mm',
                      perpendicular: bool = True,
                      region_name: str | None = None) -> FieldOfView:
    """
    Construct a :class:`~FieldOfView` from four corner coordinates.

    :param am: anteromedial corner coordinate [x, y] in mm or µm.
    :param pm: posteromedial corner coordinate [x, y] in mm or µm.
    :param pl: posterolateral corner coordinate [x, y] in mm or µm.
    :param al: anterolateral corner coordinate [x, y] in mm or µm.
    :param rotation_ml: in-plane rotation (CCW) around the ML axis, in degrees.
    :param rotation_ap: tilt around the AP axis (foreshortening), in degrees.
    :param unit: unit of the corners value
    :param perpendicular: imaging objective is perpendicular to the cranial windows. if True, skip rotation and tilt transforms.
    :param region_name: optional identifier for this FOV region.
    :returns: FieldOfView instance with corners stacked and optionally transformed.
    """
    corners = np.vstack([am, pm, pl, al])
    return FieldOfView(corners, rotation_ml, rotation_ap, unit,
                       perpendicular=perpendicular,
                       region_name=region_name)


def _validator_corners(instance, attribute, value: np.ndarray):
    if not isinstance(value, np.ndarray):
        raise TypeError(f'{attribute} should be a numpy array')
    if value.shape != (4, 2):
        raise ValueError('')


@unstable()
def _rotate_and_tilt(pts: np.ndarray, ml: float, ap: float) -> np.ndarray:
    """telecentric / orthographic imaging transformation"""
    pivot = pts.mean(axis=0)
    theta_x = np.deg2rad(ml)
    c, s = np.cos(theta_x), np.sin(theta_x)
    R = np.array([[c, -s], [s, c]])
    pts0 = (pts - pivot) @ R.T + pivot
    theta_y = np.deg2rad(ap)
    scale_x = np.cos(theta_y)
    pts1 = (pts0 - pivot) * np.array([scale_x, 1.0]) + pivot

    return pts1


@attrs.frozen
class FieldOfView:
    """Field‑Of‑View defined by its four XY corners"""

    corners: np.ndarray = attrs.field(validator=_validator_corners)
    """corner coordinates in order [AM, PM, PL, AL]. `Array[float, [4, 2]]`"""
    rotation_ml: float = attrs.field(default=0.0)
    """in-plane rotation (CCW) around the ML axis, in degrees"""
    rotation_ap: float = attrs.field(default=0.0)
    """tilt around the AP axis (foreshortening), in degrees"""
    unit: UNIT = attrs.field(default='mm', validator=attrs.validators.in_(['mm', 'um']))
    """unit of the corners value"""
    perpendicular: bool = attrs.field(default=True, kw_only=True)
    """imaging objective is perpendicular to the cranial windows. if True, skip rotation and tilt transforms"""
    region_name: str | None = attrs.field(default=None, kw_only=True)
    """optional identifier for this FOV region"""

    def __attrs_post_init__(self):
        if not self.perpendicular:
            new_corners = _rotate_and_tilt(
                self.corners,
                self.rotation_ml,
                self.rotation_ap,
            )
            object.__setattr__(self, 'corners', new_corners)

    @property
    def am(self) -> np.ndarray:
        """anteromedial corner coordinate [x, y] in mm or µm."""
        return self.corners[0]

    @property
    def pm(self) -> np.ndarray:
        """posteromedial corner coordinate [x, y] in mm or µm"""
        return self.corners[1]

    @property
    def pl(self) -> np.ndarray:
        """posterolateral corner coordinate [x, y] in mm or µm"""
        return self.corners[2]

    @property
    def al(self) -> np.ndarray:
        """anterolateral corner coordinate [x, y] in mm or µm"""
        return self.corners[3]

    @property
    def ap_distance(self) -> float:
        """span along the AP axis"""
        return float(np.ptp(self.corners[:, 1]))

    @property
    def ml_distance(self) -> float:
        """span along the ML axis"""
        return float(np.ptp(self.corners[:, 0]))

    def invert_axes(self, ap: bool = True, ml: bool = True) -> Self:
        """Return a new FOV with specified axes inverted

        :param ap: invert anterior-posterior (Y) axis if True.
        :param ml: invert medial-lateral (X) axis if True.
        """
        factors = np.array([-1. if ml else 1., -1. if ap else 1.])
        return attrs.evolve(self, corners=self.corners * factors)

    def to_um(self) -> Self:
        if self.unit == 'um':
            raise RuntimeError('unit already in um')
        return attrs.evolve(self, corners=self.corners * 1000, unit='um')

    def to_polygon(self, **kwargs) -> Polygon:
        """convert corners to a matplotlib Polygon patch"""
        idx = [0, 1, 3, 2]
        reorder = self.corners[idx]
        return Polygon(reorder, closed=True, edgecolor='r', facecolor='none', **kwargs)


def get_cellular_coordinate(neuron_idx: np.ndarray,
                            ap: np.ndarray,
                            ml: np.ndarray, *,
                            unit: UNIT = 'mm',
                            plane_index: np.ndarray | None = None) -> CellularCoordinates:
    """
    Get cellular coordinates container for doing brain mapping / topographical analysis

    :param neuron_idx: neuron index. `Array[float, N]`
    :param ap: anterior posterior coordinates. `Array[float, N]`
    :param ml: medial lateral coordinates. `Array[float, N]`
    :param unit: unit of the ap/ml value. default in `mm`
    :param plane_index: neuron's corresponding image plane. `Array[float, N]`. If None then full_zero
    :return: :class:`~CellularCoordinates`
    """
    return CellularCoordinates(neuron_idx, ap, ml, unit, plane_index)


def _validator_shape(instance: CellularCoordinates, attribute, value: np.ndarray | None):
    if value is None:
        return
    else:
        assert instance.neuron_idx.shape == value.shape


@attrs.define
class CellularCoordinates:
    """Cellular Coordinates container"""

    neuron_idx: np.ndarray
    """neuron index. `Array[float, N]`"""
    ap: np.ndarray
    """anterior posterior coordinates (default in mm). `Array[float, N]`"""
    ml: np.ndarray
    """medial lateral coordinates (default in mm). `Array[float, N]`"""
    unit: UNIT = attrs.field(default='mm', validator=attrs.validators.in_(['mm', 'um']))
    """unit of the ap/ml value"""
    plane_index: np.ndarray = attrs.field(default=None, validator=_validator_shape)
    """neuron's corresponding image plane. `Array[float, N]`"""
    value: np.ndarray | None = attrs.field(default=None, validator=_validator_shape)
    """metric (i.e., used in topographical analysis). `Array[float, N]`"""

    def __attrs_post_init__(self):
        if self.plane_index is None:
            self.plane_index = np.full_like(self.neuron_idx, 0, dtype=int)

        assert self.neuron_idx.shape == self.ap.shape == self.ml.shape == self.plane_index.shape

    def relative_origin(self, fov: FieldOfView,
                        origin: Literal['am', 'pm', 'al', 'pl'] = 'am') -> Self:
        """
        coordinates relative to :class:`~FieldOfView` origin point

        :param fov: :class:`~FieldOfView`
        :param origin: relative origin point
        :return:
        """

        factor = 1000 if self.unit == 'mm' else 1
        ap_um = self.ap * factor
        ml_um = self.ml * factor
        if fov.unit != 'um':
            fov = fov.to_um()

        orig_pt = getattr(fov, origin)
        pts = np.vstack([ml_um, ap_um]).T

        if fov.perpendicular:
            delta = orig_pt - pts  # posterior to origin
        else:
            delta_pts = orig_pt - _rotate_and_tilt(
                pts,
                ml=fov.rotation_ml,
                ap=fov.rotation_ap,
            )
            delta = delta_pts

        ml_new, ap_new = delta[:, 0], delta[:, 1]
        return attrs.evolve(self, ml=ml_new, ap=ap_new, unit='um')

    def with_value(self, value: np.ndarray) -> Self:
        """assign value foreach neuron"""
        return attrs.evolve(self, value=value)

    def with_masking(self, mask: np.ndarray) -> Self:
        """do neuronal selection by bool masking
        :param mask: `Array[bool, N]`
        """
        return attrs.evolve(
            self,
            neuron_idx=self.neuron_idx[mask],
            ap=self.ap[mask],
            ml=self.ml[mask],
            plane_index=self.plane_index[mask],
            value=None if self.value is None else self.value[mask]
        )
