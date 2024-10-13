import numpy as np
from numba import jit

from neuralib.util.segments import SegmentLike
from neuralib.util.unstable import unstable

__all__ = ['running_mask1d',
           'jump_mask2d']


def running_mask1d(time: np.ndarray,
                   velocity: np.ndarray,
                   *,
                   threshold: float = 5,
                   merge_gap: float = 0.5,
                   minimal_time: float = 1) -> np.ndarray:
    """
    Running epoch mask for linear (1-dimension) locomotion.

    Refer to Zaremba et al., 2016. Nature Neuroscience.
    Selection based on criterion:

    - Forward locomotion
    - Velocity > threshold (default greater than 5 cm/s)
    - Merge the gap period (default 0.5 sec, if animal accidentally slow down)
    - Each segmented epoch need to longer than (default is 1 second)

    `Dimension parameters`:

        P = Number of position points


    :param time: 1D time array of the position. `Array[float, P]`
    :param velocity: 1D velocity array. `Array[float, P]`
    :param threshold: Velocity threshold in cm/s
    :param merge_gap: Merge gap in seconds (i.e., animal accidentally slow down in a short period of time)
    :param minimal_time: Epoch minimal time in seconds
    :return: Running epoch mask. `Array[bool, P]`
    """
    run = (velocity > threshold).astype(int)

    # epoch_list: [0*epoch0, 1*epoch1, 2*epoch2, ...]
    epoch = np.cumsum(abs(np.diff(run, prepend=run[0])))

    # measure even numbers of epoch index are gaps
    if velocity[0] >= threshold:
        epoch += 1

    # merge gap
    for i in range(2, np.max(epoch) + 1, 2):
        t = time[epoch == i]
        dt = np.max(t) - np.min(t)
        if dt < merge_gap:
            epoch[epoch == i] += 1
            epoch[epoch == i + 1] += 2

    # minimal time
    for i in range(1, np.max(epoch) + 1, 2):  # exclude 0 due to merge special case
        tt = time[epoch == i]
        if len(tt) > 0:  # i is not existed in ep_idx
            td = np.max(tt) - np.min(tt)
            if td < minimal_time:
                epoch[epoch == i] += 1

    return np.asarray(epoch % 2 != 0)


# noinspection PyTypeChecker
@unstable()
@jit(nopython=True)
def jump_mask2d(time: np.ndarray,
                xy: np.ndarray,
                jump_size: float,
                max_duration: float = 0.1) -> tuple[np.ndarray, SegmentLike]:
    """
    Find jump epoch in 2D xy locomotion

    :param time: A 1D numpy array of time points corresponding to each xy coordinate. `Array[bool, N]`
    :param xy: A 2D numpy array of x and y coordinates representing positions. `Array[float, [N, 2]]`
    :param jump_size: A float representing the minimum jump distance to be considered significant between consecutive points.
        Should be expressed in the same coordinates as xy
    :param max_duration: the maximum duration (in the same units as t) of a jump.
        If a jump lasts longer the duration, it will not be removed (as it may not actually be a jump)
    :return: A tuple containing:

        - A boolean numpy array indicating which points are part of a detected jump segment. `Array[bool, N]`

        - A list of segments, where each segment is represented by a list of two indices `[start_idx, end_idx]`.
    """
    valid = np.nonzero(~np.isnan(xy[:, 0]))[0]
    ret = np.zeros(xy.shape[0], dtype=np.bool_)
    seg = [[np.int64(x) for x in range(0)]]

    if len(valid) < 2:
        return ret, seg  # without detect

    found_jump = False
    last_idx = valid[0]

    for idx in valid[1:]:

        dt = time[idx] - time[last_idx]
        dxy = np.sqrt(np.sum((xy[idx] - xy[last_idx]) ** 2))

        if not found_jump and (dxy / (idx - last_idx)) > jump_size:
            found_jump = True

        if found_jump and (dxy <= jump_size or dt >= max_duration):

            if dt < max_duration:
                ret[last_idx:idx] = 1
                seg.append([last_idx, idx])

            found_jump = False

        if not found_jump:
            last_idx = idx

    return ret, seg[1:]
