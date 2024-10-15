from __future__ import annotations

import numpy as np

from neuralib.util.deprecation import deprecated_class, deprecated_func


@deprecated_class(new='neuralib.util.segments.*', removal_version="0.4.0")
class Segment:
    """Segment container class"""

    def __init__(self,
                 data: Segment | np.ndarray,
                 copy: bool = True):

        if isinstance(data, Segment):
            if copy:
                self._data = data._data.copy()
            else:
                self._data = data._data
        else:
            self._data = check_segments(data, copy)

    def __repr__(self):
        """Return string representation of Segment object."""
        return f'Segment(" + {repr(self._data)} + ")'

    __str__ = __repr__

    def __len__(self):
        """Return the number of segments in the container."""
        return int(self._data.shape[0])

    def __getitem__(self, item: slice | int) -> Segment:
        """

        :param item: slice or index
        :return:
        """
        return Segment(self._data[item, :])

    def __iter__(self):
        """Iterate through segments in container."""
        idx = 0
        while idx < self._data.shape[0]:
            yield self._data[idx, 0], self._data[idx, 1]
            idx += 1

    @classmethod
    def from_logical(cls,
                     y: np.ndarray,
                     x: np.ndarray | None = None,
                     interpolate=False) -> Segment:
        """Construct Segment from logical vector

        :param y: 1d logical array
            Any sequence of True values that is flanked by False values is
            converted into a segment
        :param x: 1d array like, optional
            The segment indices from y will be used to index into x
        :param interpolate: bool, optional
            if true, segments of duration 0 are extent to have a minimal duration

        """
        y = np.where(y, 1, 0).ravel()

        if len(y) == 0 or np.all(y == 0):
            return cls(np.zeros((0, 2), dtype=np.int64))

        offset = 0
        if interpolate:
            offset = 0.5

        d = np.diff(np.concatenate(([0], y, [0])))
        segstart = np.nonzero(d[0:-1] == 1)[0] - offset
        segend = np.nonzero(d[1:] == -1)[0] + offset

        seg = np.vstack((segstart, segend)).T

        if x is not None:
            if interpolate:
                from scipy.interpolate import interp1d
                seg = interp1d(
                    np.arange(len(y)),
                    x,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(x[0], x[-1]),
                )(seg)
            else:
                seg = x[seg]

        return cls(seg)

    @property
    def start(self) -> np.ndarray:
        """Get/Set vector of segment start values
        """
        return self._data[:, 0].copy()

    @property
    def stop(self) -> np.ndarray:
        return self._data[:, 1]


@deprecated_func(new='neuralib.util.segments.*', removal_version="0.4.0")
def check_segments(x: np.ndarray, copy=False) -> np.ndarray:
    """Convert to segment array.

    :param x: 1d array-like or (n,2) array-like
    :param copy: the output will always be a copy of the input
    :return: (n,2) array
    """
    try:
        x = np.array(x, copy=copy)
    except TypeError:
        raise ValueError("Cannot convert data to numpy array")

    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("Values are not real numbers")

    # The array has to have two dimensions of shape(X,2), where X>=0.
    # As a special case, a one dimensional vector of at least length two is considered
    # a valid list of segments, e.g. when data is specified as a list [0,2,3].

    if x.shape == (0,):
        x = np.zeros([0, 2])
    elif x.ndim == 1 and len(x) > 1:
        x = np.vstack((x[0:-1], x[1:])).T
    elif x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("Incorrect array size")

    # Negative duration segments are not allowed.
    if np.any(np.diff(x, axis=1) < 0):
        raise ValueError("Segment durations cannot be negative")

    return x
