import numpy as np

from neuralib.signal.segement import Segment

__all__ = [
    'interpolate_gaps'
]


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
