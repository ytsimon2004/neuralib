import numpy as np

__all__ = ['interp_timestamp',
           'interp1d_nan']


def interp_timestamp(timestamp: np.ndarray,
                     t0: float,
                     t1: float,
                     sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Create interpolated time and value(01) array from event timestamp

    `Dimension parameters`:

        T = number of timestamp

        N' = number of interpolated sample points

    :param timestamp: Array of event timestamps in seconds. `Array[float, T]`
    :param t0: Start time in seconds
    :param t1: End time in seconds
    :param sampling_rate: Sampling rate in Hz
    :return: Tuple of time array (`Array[float, N']`) and event array (`Array[float, N']`)
    """

    n = int((t1 - t0) * sampling_rate)
    tt = np.linspace(t0, t1, num=n)

    sig = np.zeros(n, dtype=np.int8)
    indices = ((timestamp - t0) * sampling_rate).astype(int)
    indices = indices[(indices >= 0) & (indices < n)]
    sig[indices] = 1

    return tt, sig


def interp1d_nan(arr: np.ndarray) -> np.ndarray:
    """
    Interpolates missing values (NaNs) in a 1-dimensional NumPy array.

    :param arr: Input array with potential NaNs. `Array[float, N]`
    :return: Array with NaNs interpolated. `Array[float, N]`
    """
    n = len(arr)
    x = np.arange(n)
    isnan = np.isnan(arr)

    if isnan.any():
        if isnan.all():
            raise ValueError('Input array contains only NaNs')

        arr_nonan = arr[~isnan]
        x_nonan = x[~isnan]
        arr[isnan] = np.interp(x[isnan], x_nonan, arr_nonan)

    arr_clean = arr[~np.isnan(arr)]

    return arr_clean
