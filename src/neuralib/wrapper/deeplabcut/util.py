import numpy as np
from numba import jit

from neuralib.util.segement import Segment

__all__ = ['compute_velocity',
           'remove_jumps',
           'interpolate_gaps']


def compute_velocity(xy: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """Compute velocity

    :param xy: (n,2) coordinates
    :param dx: time period between samples (for smoothing)
    :return:
        (n,) complex array
        velocity as a complex number. To compute speed, use np.abs(vel).
        To compute movement direction, use np.angle(vel).
    """
    return np.gradient(xy[:, 0], dx) + np.gradient(xy[:, 1], dx) * 1.0j


@jit(nopython=True)
def remove_jumps(t: np.ndarray,
                 xy: np.ndarray,
                 jump_size: float,
                 duration: float = 0.1) -> tuple[np.ndarray, list[list[int]]]:
    """

    :param t: (n, ) time vector
    :param xy: (n, 2) xy position coordinates
    :param jump_size: the maximum shift in position that is not yet considered a jump. Should be expressed in the same coordinates as xy
    :param duration: the maximum duration (in the same units as t) of a jump. If a jump lasts longer than maxduration, it will not be
        removed (as it may not actually be a jump)
    :return:
        xy : (n,2) array
            corrected x,y coordinated (positions during jumps are set to np.nan)
        jumps : sequence
            a list of [start,end] indices for each detected jump
    """
    xy = xy.copy()

    # only work with the valid coordinates
    valid = np.flatnonzero(np.logical_not(np.isnan(xy[:, 0])))

    if len(valid) < 2:
        return xy, [[np.int64(x) for x in range(0)]]

    last_valid = valid[0]
    found_jump = False
    jumps = [[np.int64(x) for x in range(0)]]

    # loop through all valid coordinates
    for idx in valid[1:]:

        # compute time since and distance to last valid coordinate
        dx = t[idx] - t[last_valid]
        dy = np.sqrt(np.sum((xy[idx] - xy[last_valid]) ** 2))

        if not found_jump and dy / (idx - last_valid) > jump_size:
            # we detected a jump larger than the threshold
            found_jump = True

        if found_jump and (dy <= jump_size or dx >= duration):
            # and now we detected a jump back within the maximum allowed duration
            # or this a long jump
            if dx < duration:
                xy[last_valid:idx, :] = np.NaN
                jumps.append([last_valid, idx])

            found_jump = False

        if not found_jump:
            last_valid = idx

    return xy, jumps


def interpolate_gaps(t: np.ndarray,
                     xy: np.ndarray,
                     interp_gap: float) -> np.ndarray:
    """Interpolate over gaps in position record.

    :param t: time vector
    :param xy: x,y coordinates
    :param interp_gap: maximum duration of a gap that should be interpolated over

    :return xy: (n,2) array
        corrected x,y coordinates
    """
    from scipy.interpolate import interp1d

    xy = xy.copy()

    # small gap interpolation
    invalid = np.isnan(xy[:, 0])
    valid = np.logical_not(invalid)
    # create index segments for all invalid gaps
    invalid_segments = Segment.from_logical(invalid)
    # select small gaps
    invalid_segments = invalid_segments[
        (t[invalid_segments.stop] - t[invalid_segments.start]) < interp_gap
        ]
    if len(invalid_segments) > 0:
        # construct index vector for all small gaps
        invalid_indices = np.concatenate(
            [np.arange(start, stop + 1) for start, stop in invalid_segments]
        )
        # interpolate small gaps
        fcn = interp1d(
            t[valid],
            xy[valid, :],
            kind="linear",
            bounds_error=False,
            axis=0,
            assume_sorted=True,
        )
        xy[invalid_indices, :] = fcn(t[invalid_indices])

    return xy
