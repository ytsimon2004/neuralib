import numpy as np


def filter_outliers(x, y, filter_window=15, baseline_window=50, max_spike=25, max_diff=2):
    """from ``facemap.utils``"""
    # remove frames with large jumps
    x_diff = np.abs(np.append(np.zeros(1, ), np.diff(x), ))
    y_diff = np.abs(np.append(np.zeros(1, ), np.diff(y), ))
    replace_inds = np.logical_or(x_diff > max_diff, y_diff > max_diff)
    x[replace_inds] = np.nan
    y[replace_inds] = np.nan

    # remove frames with large deviations from baseline
    x_baseline = nanmedian_filter(x, baseline_window)
    y_baseline = nanmedian_filter(y, baseline_window)
    replace_inds = np.logical_or(
        np.abs(x - x_baseline) > max_spike, np.abs(y - y_baseline) > max_spike
    )
    x[replace_inds] = np.nan
    y[replace_inds] = np.nan
    replace_inds = np.isnan(x)

    # filter x and y
    x_filt = nanmedian_filter(x, filter_window)
    y_filt = nanmedian_filter(y, filter_window)

    # this in theory shouldn't add more frames
    replace_inds = np.logical_or(replace_inds, np.isnan(x_filt))
    ireplace = np.nonzero(replace_inds)[0]

    # replace outlier frames with median
    if len(ireplace) > 0:
        # good indices
        iinterp = np.nonzero(np.logical_and(~replace_inds, ~np.isnan(x_filt)))[0]
        x[replace_inds] = np.interp(ireplace, iinterp, x_filt[iinterp])
        y[replace_inds] = np.interp(ireplace, iinterp, y_filt[iinterp])

    return x, y


def nanmedian_filter(x, win=7):
    """nanmedian filter array along last axis"""
    nt = x.shape[-1]
    # pad so that x will be divisible by win
    pad = (win - (nt + 2 * (win // 2)) % win) % win
    xpad = np.pad(x, (win // 2, win // 2 + win + pad), mode="edge")
    xmed = np.zeros_like(x)
    for k in range(win):
        xm = np.nanmedian(xpad[k: k - win].reshape(-1, win), axis=-1)
        xmed[..., k::win] = xm[: len(np.arange(k, nt, win))]
    return xmed
