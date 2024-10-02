import numpy as np

__all__ = ['running_mask1d']


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
