import numpy as np

from neuralib.ephys.kilosort.result import KilosortResult

__all__ = [
    'next_cluster_id',
    'nearby_channels',
    'template_peak_channel',
]


def next_cluster_id(ks_data: KilosortResult) -> int:
    return np.max(ks_data.spike_cluster) + 1


def nearby_channels(ks_data: KilosortResult,
                    channel_index: int, *,
                    distance: float | tuple[float, float] = None,
                    count: int = None) -> np.ndarray:
    """
    
    :param ks_data: 
    :param channel_index: channel index
    :param distance: a distance in um, or tuple of `(x, y)`
    :param count: number of returned channels.
    :return: 
    """
    pos = ks_data.channel_pos
    c = pos[int(channel_index)]
    dp = pos - c
    dx = np.abs(dp[:, 0])
    dy = np.abs(dp[:, 1])
    d = np.sqrt(dx ** 2 + dy ** 2)
    i = np.argsort(d)
    d = d[i]
    dx = dx[i]
    dy = dy[i]

    if distance is not None:
        if isinstance(distance, (int, float)):
            i = i[d <= distance]
        else:
            _x, _y = distance
            i = i[(dx <= _x) & (dy <= _y)]

    if count is not None:
        i = i[:count]

    assert np.any(i == channel_index)
    return i


def template_peak_channel(template: np.ndarray) -> int | np.ndarray:
    """

    :param template: Array[float, [S, C]] or Array[float, [U, S, C]]
    :return: channel index (array if template.ndim==3)
    """
    if template.ndim == 2:  # (S, C)
        return int(np.argmax(np.max(template, axis=0) - np.min(template, axis=0)))
    elif template.ndim == 3:  # (U, S, C)
        return np.argmax(np.max(template, axis=1) - np.min(template, axis=1), axis=1)
    else:
        raise ValueError('wrong dimension')
