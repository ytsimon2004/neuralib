from typing import NamedTuple

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from typing_extensions import Self

from neuralib.util.segments import segment_epochs, segment_duration, segment_contains
from neuralib.util.unstable import unstable

__all__ = [
    'CircularPosition',
    'interp_pos1d',
    #
    'speed_2d',
    'direction_2d',
    'interp_gap2d'
]


class CircularPosition(NamedTuple):
    """
    Position information in circular environment.

    `Dimension parameters`:

        P = Number of position points (after interpolation)

        T = Number of trial(lap)

    """

    t: np.ndarray
    """1D time array in s. `Array[float, P]`"""

    p: np.ndarray
    """1D position array in cm. `Array[float, P]`"""

    d: np.ndarray
    """1D displacement array in cm. `Array[float, P]`"""

    v: np.ndarray
    """1D velocity array in cm/s. `Array[float, P]`"""

    trial_time_index: np.ndarray
    """1D time index for every trial. `Array[int, T]`"""

    def with_time_range(self, t0: float, t1: float) -> Self:
        """With specific time range

        :param t0: time start
        :param t1: time end
        :return: A new instance of the class with updated subset of attributes (t, p, d, v) based on the time range.
        """
        tx0 = self.t >= t0
        tx1 = self.t <= t1
        tx = np.logical_and(tx0, tx1)

        ix0 = np.nonzero(tx0)[0][0]  # start index for time
        ix1 = np.nonzero(~tx1)[0][0]  # end index for time
        ix = np.logical_and(self.trial_time_index >= ix0, self.trial_time_index <= ix1)

        return self._replace(
            t=self.t[tx],
            p=self.p[tx],
            d=self.d[tx],
            v=self.v[tx],
            trial_time_index=self.trial_time_index[ix]
        )

    def with_run_mask1d(self, **kwargs) -> Self:
        """With only the running epoch. **Note that shape will be changed (stationary epoch excluded)**

        :param kwargs: Additional keyword arguments to pass to running_mask1d function.
        :return: A new instance of the class with updated subset of attributes (t, p, d, v) based on the mask generated.
        """
        from .epoch import running_mask1d
        x = running_mask1d(self.t, self.v, **kwargs)
        return self._replace(
            t=self.t[x],
            p=self.p[x],
            d=self.d[x],
            v=self.v[x],
        )

    @property
    def trial_array(self) -> np.ndarray:
        """Trial number array as same shape as *P*

        :return: `Array[int, P]`
        """
        ret = np.zeros_like(self.t, int)
        ret[self.trial_time_index] = 1
        return np.cumsum(ret)


def interp_pos1d(time: np.ndarray,
                 pos: np.ndarray,
                 *,
                 sampling_rate: float = 1000,
                 norm_max_value: float = 150,
                 remove_nan: bool = True,
                 renew_trial_value: float = -100) -> CircularPosition:
    """
    Interpolate the raw position data

    `Dimension parameters`:

        P = Number of position points (Raw)

    :param time: An array of time stamps corresponding to positional data. `Array[float, P]`.
    :param pos: An array of positional data (a.u, encoder or other hardware readout). `Array[float, P]`.
    :param sampling_rate: The rate at which data points should be sampled, defaults to 1000.
    :param norm_max_value: The value by which to normalize the positional data, defaults to 150 (cm).
    :param remove_nan: Flag indicating whether to remove NaN values from interpolated data, defaults to True.
    :param renew_trial_value: The value used to detect the overflow index which signifies trial renewals, defaults to -100.
    :return: ``CircularPosition``
    """

    overflow_idx = np.nonzero(np.diff(pos) < renew_trial_value)[0]

    # Get the position(p), displacement(d)
    if len(overflow_idx) == 0:  # not more than a trial
        p = pos / np.max(pos)
        d = pos / np.max(pos)
    else:
        p = np.zeros_like(pos, float)
        d = np.zeros_like(pos, float)
        mode = stats.mode(pos[overflow_idx], keepdims=True).mode.astype(int)

        # fix the first trial
        p[:overflow_idx[0] + 1] = pos[:overflow_idx[0] + 1] / pos[overflow_idx[0]]
        d[:overflow_idx[0] + 1] = pos[:overflow_idx[0] + 1] / pos[overflow_idx[0]]
        lap_counter = 1

        # loop through rest of trials
        for lap_start, lap_stop in zip(overflow_idx[:-1], overflow_idx[1:]):
            p[lap_start + 1:lap_stop + 1] = pos[lap_start + 1:lap_stop + 1] / pos[lap_stop]
            d[lap_start + 1:lap_stop + 1] = pos[lap_start + 1:lap_stop + 1] / pos[lap_stop] + lap_counter
            lap_counter += 1

        # fix the last lap
        p[overflow_idx[-1] + 1:] = p[overflow_idx[-1] + 1:] / mode
        d[overflow_idx[-1] + 1:] = p[overflow_idx[-1] + 1:] / mode + lap_counter

    # actual length
    p *= norm_max_value
    d *= norm_max_value

    # interpolate
    t0 = np.min(time)
    t1 = np.max(time)
    tt = np.linspace(t0, t1, int((t1 - t0) * sampling_rate))
    pp = interp1d(time, p, kind='nearest', copy=False, bounds_error=False, fill_value=np.nan)(tt)
    dd = interp1d(time, d, kind='linear', copy=False, bounds_error=False, fill_value='extrapolate')(tt)
    vv = np.diff(dd, prepend=dd[0]) * sampling_rate

    # remove nan from position_value (from interpolation boundary error)
    if remove_nan:
        x = ~np.isnan(pp)
        tt = tt[x]
        pp = pp[x]
        dd = dd[x]
        vv = vv[x]

    # time index foreach trial
    if len(overflow_idx) == 0:
        trial_time_index = np.array([0])
    else:
        trial_time_index = np.zeros_like(overflow_idx, int)
        for i, t in enumerate(time[overflow_idx]):
            # shape of t changed after interpolation(tt), find the minimal tt right after the t
            trial_time_index[i] = np.nonzero(tt >= t)[0][0]

    return CircularPosition(tt, pp, dd, vv, trial_time_index)


# =========== #
# 2D Movement #
# =========== #

def _complex_2d(xy: np.ndarray, dt: float = 1.0) -> np.ndarray:
    return np.gradient(xy[:, 0], dt) + np.gradient(xy[:, 1], dt) * 1.0j


def speed_2d(xy: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Compute speed 2D movement

    :param xy: 2D array coordinates in xy. `Array[float, [N, 2]]`
    :param dt: Time period between samples (for smoothing)
    :return: Speed. `Array[float, N]`
    """
    return np.abs(_complex_2d(xy, dt))


def direction_2d(xy: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Compute direction 2D movement

    :param xy: 2D array coordinates in xy. `Array[float, [N, 2]]`
    :param dt: Time period between samples (for smoothing)
    :return: Movement direction. `Array[float, N]`
    """
    return np.angle(_complex_2d(xy, dt), deg=True)


@unstable()
def interp_gap2d(t: np.ndarray,
                 xy: np.ndarray,
                 duration: float) -> np.ndarray:
    """
    Interpolate over gaps in position record.

    :param t: Time Vector. `Array[float, N]`
    :param xy: 2D array coordinates in xy. `Array[float, [N, 2]]`
    :param duration: Maximum duration of a gap that should be interpolated over. same unit as ``t``
    :return: Corrected x,y coordinates. `Array[float, [N, 2]]`
    """
    invalid = np.isnan(xy[:, 0])
    valid = ~invalid

    # select small gaps
    invalid_seg = segment_epochs(invalid, t)
    invalid_seg = invalid_seg[segment_duration(invalid_seg) < duration]

    if invalid_seg.shape[0] > 0:
        print(f'{len(invalid_seg)} segments invalid')
        invalid_indices = np.nonzero(segment_contains(invalid_seg, t))[0]

        f = interp1d(
            t[valid],
            xy[valid, :],
            kind="linear",
            bounds_error=False,
            axis=0,
            assume_sorted=True,
        )
        xy[invalid_indices, :] = f(t[invalid_indices])

    return xy
