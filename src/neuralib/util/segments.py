from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

__all__ = [
    'is_sorted',
    'segment_mask',
    'segment_epochs',
    'has_gap',
    'segment_gap',
    'as_segment',
    'segment_range',
    'segment_duration',
    'segment_at_least_duration',
    'segment_expand_duration',
    'segment_universe',
    'segment_flatten',
    'segment_invert',
    'segment_intersection',
    'segment_union',
    'segment_diff',
    'segment_contains',
    'segment_index',
    'segment_overlap',
    'segment_overlap_index',
    'segment_join',
    'segment_join_index',
    'segment_map',
    'segment_group_map',
    'segment_sample',
    'segment_bins',
]


Segment = np.ndarray  # (N, 2) value array ([(start, stop)]), as a segment.
SegmentLike = Segment | tuple[float, float] | list[tuple[float, float]]
SegmentGroup = np.ndarray  # (N,) int array, where a unique value indicate a unique group/segment.


def is_sorted(a: np.ndarray, strict=False) -> bool:
    """
    [reference](https://stackoverflow.com/a/47004507)

    :param a:
    :param strict:
    :return:
    """
    if strict:
        return np.all(a[:-1] < a[1:])
    else:
        return np.all(a[:-1] <= a[1:])


def segment_mask(x: np.ndarray, t: np.ndarray = None,
                 duration: float = None,
                 merge: float = None) -> SegmentGroup:
    """segmenting an array that the value is long enough.

    :param x: (N,) int-cast-able array.
    :param t: (N,) T-value array
    :param duration: T. the minimal True duration, duration smaller than value will set to even value.
        If using negative value, then duration larger than value will set to even value.
    :param merge: T. the minima False duration, duration smaller than value will set to odd value.
    :return: (N,) int array, where even value indicate False, odd value indicate True
    """
    if x.ndim != 1:
        raise ValueError('x ndim not 1')
    if t is None:
        t = np.arange(len(x))
    elif not is_sorted(t, strict=True):
        raise ValueError('t not sorted')

    if x.shape != t.shape:
        raise ValueError('t.shape != x.shape')
    if len(x) == 0:
        return np.array([], dtype=int)

    edge = np.sign(np.diff(x.astype(int), prepend=0))
    s = np.cumsum(np.abs(edge))
    sx = s[-1]

    if merge is not None:
        for i in range(2, sx + 1, 2):
            if len(d := t[s == i]) > 0:
                if d[-1] - d[0] <= merge:
                    s[s == i - 1] = i + 1
                    s[s == i] = i + 1

    if duration is not None:
        if duration > 0:
            for i in range(1, sx + 1, 2):
                if len(d := t[s == i]) > 0:
                    if d[-1] - d[0] <= duration:
                        s[s == i] = i - 1
        elif duration < 0:
            for i in range(1, sx + 1, 2):
                if len(d := t[s == i]) > 0:
                    if d[-1] - d[0] >= duration:
                        s[s == i] = i - 1
    return s


def segment_epochs(x: np.ndarray, t: np.ndarray = None,
                   duration: float = None,
                   merge: float = None) -> Segment:
    """

    :param x: (N,) int-cast-able array.
    :param t: (N,) T-value array.
    :param duration:
    :param merge:
    :return: (M, 2) T-value array. If *t* is None, return an N-value index array
    """
    if t is None:
        t = np.arange(len(x))

    if x.shape != t.shape:
        raise ValueError('t.shape != x.shape')

    if len(x) == 0:
        return np.zeros((0, 2), dtype=t.dtype)

    g = segment_mask(x, t, duration, merge)

    ret = []
    for gg in range(1, np.max(g) + 1, 2):
        if len(i := np.nonzero(g == gg)[0]):
            if t is None:
                ret.append((i[0], i[-1]))
            else:
                ret.append((t[i[0]], t[i[-1]]))
    return np.array(ret)


def has_gap(y: np.ndarray, gap: float) -> bool:
    """

    :param y: (N,) V-value array
    :param gap: V
    :return:
    """
    return np.any(np.abs(np.diff(y)) > gap)


def segment_gap(x: np.ndarray, gap: float) -> SegmentGroup:
    """segmenting an array that cut at the place which the difference of nearby value larger than *gap*.

    `min{|ai-aj|} > gap` for any value `ai` from segment `si`, any value `aj` from segment `sj`, `si != sj`.

    :param x: (N,) V-value array
    :param gap: V
    :return: (N,) int-group array
    """
    if len(x) == 0:
        raise ValueError('empty array')
    elif len(x) == 1:
        return np.array([0])
    else:
        return np.cumsum(np.abs(np.diff(x, prepend=x[0])) > gap)

def as_segment(segs: SegmentLike) -> Segment:
    ret = np.atleast_2d(segs)
    if ret.ndim != 2 or ret.shape[1] != 2:
        raise ValueError(f'not a (N, 2) segment array : {ret.shape}')
    return ret


def segment_range(segs: SegmentLike) -> np.ndarray:
    """change [(start, stop)] to [(start, duration)]

    :param segs: (N, 2) segments
    :return: (N, 2) array
    """
    a = as_segment(segs)
    r = a.copy()
    r[:, 1] = a[:, 1] - a[:, 0]
    return r


def segment_duration(segs: SegmentLike) -> np.ndarray:
    """

    :param segs: (N, 2) T-value segments
    :return: (N,) T-value array
    """
    a = as_segment(segs)

    if a.ndim == 1 and len(a) == 2:
        a = np.array([a])

    if a.ndim != 2:
        raise ValueError(f'not a (N, 2) segment array : {a.shape}')

    return a[:, 1] - a[:, 0]


def segment_at_least_duration(segs: SegmentLike, duration: float) -> np.ndarray:
    """

    :param segs: (N, 2) T-value segments
    :param duration: T-value
    :return: (N, 2) T-value segments
    """
    a = as_segment(segs).copy()
    dur = (a[:, 1] - a[:, 0]) - duration
    ext = np.where(dur > 0, dur / 2, 0)
    a[:, 0] -= ext
    a[:, 1] += ext

    return a


def segment_expand_duration(segs: SegmentLike, duration: float | tuple[float, float] | np.ndarray) -> np.ndarray:
    """

    :param segs: (N, 2) T-value segments
    :param duration: T-value scalar, tuple(prepend:T, append:T), or (N, 2?) array
    :return: (N, 2) T-value segments
    """
    a = as_segment(segs).copy()

    if isinstance(duration, tuple):
        if len(duration) != 2:
            raise ValueError()

        ext = duration

    elif isinstance(duration, np.ndarray):
        if duration.ndim == 1:
            ext = duration / 2
            ext = ext, ext
        elif duration.ndim == 2:
            ext = duration
        else:
            raise ValueError()

    else:
        ext = duration / 2, duration / 2

    a[:, 0] -= ext[0]
    a[:, 1] += ext[1]

    return a


def segment_universe() -> Segment:
    """

    :return: segment of [-inf, inf]
    """
    return np.array([[-np.Inf, np.Inf]])


def segment_flatten(a: SegmentLike, closed=True) -> Segment:
    """sorting and remove overlapped segments.

    :param a: (N, 2) T-value segments
    :param closed: Is segment a closed on right side?
    :return: (M, 2) T-value segments
    """
    a = as_segment(a)

    if len(a) == 0:
        return a

    if not is_sorted(a[: 0]):
        a = a[np.argsort(a[:, 0])]

    if is_sorted(a.ravel(), strict=True):
        return a

    a = a.copy()
    i = 1
    while i < len(a):
        p = a[i - 1, 1]
        q = a[i, 0]
        if p >= q if closed else p > q:
            a[i - 1, 1] = max(a[i, 1], a[i - 1, 1])
            a = np.delete(a, i, axis=0)
        else:
            i += 1

    return a


def segment_invert(a: SegmentLike) -> Segment:
    """

    :param a: (N, 2) T-value segments
    :return:  (N+1, 2) T-value segments
    """
    return _segment_invert(segment_flatten(a))


def _segment_invert(a: Segment) -> Segment:
    a = np.concatenate([[-np.Inf], a.ravel(), [np.Inf]]).reshape((-1, 2))

    d = []
    if np.all(np.isinf(a[0])):
        d.append(0)
    if np.all(np.isinf(a[-1])):
        d.append(-1)

    return np.delete(a, d, axis=0)


def segment_intersection(a: SegmentLike, b: SegmentLike) -> Segment:
    """

    :param a: (A, 2) T-value segments
    :param b: (B, 2) T-value segments
    :return: (C, 2) T-value segments
    """
    a = segment_flatten(a)
    b = segment_flatten(b)
    ret = []

    i = 0
    j = 0
    while i < len(a) and j < len(b):
        p, q = a[i]
        s, t = b[j]
        if p <= s:
            if s < q:  # p <= s < {q  t}
                ret.append((s, min(q, t)))
        else:
            if p < t:  # s < p < {q  t}
                ret.append((p, min(q, t)))

        if q <= t:
            i += 1
        else:
            j += 1

    if len(ret) == 0:
        return np.zeros((0, 2), a.dtype)

    return np.vstack(ret)


def segment_union(a: SegmentLike, b: SegmentLike, gap: float = 0) -> Segment:
    """

    :param a: (A, 2) T-value segments
    :param b: (B, 2) T-value segments
    :param gap: T value
    :return: (C, 2) T-value segments
    """
    a = segment_flatten(a)
    b = segment_flatten(b)

    if len(a) == 0:
        return b
    elif len(b) == 0:
        return a

    ret = []

    i = 0
    j = 0
    while i < len(a) and j < len(b):
        p, q = a[i]
        s, t = b[j]
        if q + gap < s:  # p < q+g < s < t
            ret.append((p, q))
            i += 1
        elif t + gap < p:  # s < t+g < p < q
            ret.append((s, t))
            j += 1
        elif p <= s:  # p < s < {q, t}
            ret.append((p, max(q, t)))
            i += 1
        else:  # s < p < {q, t}
            ret.append((s, max(q, t)))
            j += 1

    while i < len(a):
        ret.append(tuple(a[i]))
        i += 1
    while j < len(b):
        ret.append(tuple(b[j]))
        j += 1
    return segment_flatten(np.vstack(ret))


def segment_diff(a: SegmentLike, b: SegmentLike) -> Segment:
    """

    :param a: (A, 2) T-value segments
    :param b: (B, 2) T-value segments
    :param gap: T value
    :return: (C, 2) T-value segments
    """
    a = segment_flatten(a)
    b = segment_flatten(b)
    return segment_intersection(a, _segment_invert(b))


def segment_contains(segs: SegmentLike, t: np.ndarray) -> np.ndarray:
    """
    whether *t* in the segments.

    :param segs: (N, 2) T-value segments
    :param t: (R,) T-value array
    :return: (R,) bool array
    """
    segs = segment_flatten(segs)
    x1 = np.less_equal.outer(segs[:, 0], t)  # (N, T), s[0] <= t
    x2 = np.greater_equal.outer(segs[:, 1], t)  # (N, T), s[1] >= t
    xx = np.logical_and(x1, x2)
    return np.logical_or.reduce(xx, axis=0)


def segment_index(segs: SegmentLike, t: np.ndarray) -> np.ndarray:
    """find the index of *segs* where *t* located

    ::

        segs : ---[---]---[---)[---]---
        ret  :  -1  0   -2  1    2   -4

    The following code are true

    .. code-block :: python

        I = segment_index(S, T)
        for i in np.nonzero(I >= 0)[0]:
            assert S[I[i], 0] <= t[i] <= S[I[i], 1]

        for i in np.nonzero(I < 0)[0]:
            if -I[t] - 1 == 0:
                assert t[i] < S[-I[t] - 1, 0]
            elif -I[t] - 1 == len(S):
                assert S[-I[i] - 2, 1] < t[i]
            else:
                assert S[-I[i] - 2, 1] < t[i] < S[-I[t] - 1, 0]

    :param segs: (N, 2) T-value segments
    :param t: (R,) T-value array
    :return: (R,) N-value index array
    :raise ValueError: *segs* has overlapped segments or not sorted
    """
    if len(segment_flatten(segs, closed=False)) != len(segs):
        raise ValueError()

    f = segs.ravel()
    if not is_sorted(f):
        raise ValueError()

    i = np.searchsorted(f, t, 'right')  # Array[int, T]
    pos = i % 2 == 1
    neg = i % 2 == 0
    i[pos] = (i[pos] - 1) // 2
    i[neg] = -i[neg] // 2 - 1
    return i


def segment_overlap(segs: SegmentLike, t: Segment, mode: Literal['in', 'out', 'overlap']) -> np.ndarray:
    """
    * mode == 'in' ::

        returns = [∃ s in S st. t ⊂ s | for t in T]

    * mode == 'out' ::

        returns = [∃ s in S st. s ⊂ t | for t in T]

    * mode == 'overlap' ::

        returns = [∃ s in S st. s ⋂ t ≠ ∅ | for t in T]

    :param segs: (N, 2) T-value segments
    :param t:  (R, 2) T-value segments
    :param mode:
    :return: (R,) bool array
    """
    segs = segment_flatten(segs)
    msk = _segment_overlap(segs, t, mode)
    return np.logical_or.reduce(msk, axis=0)


def segment_overlap_index(segs: SegmentLike, t: Segment, mode: Literal['in', 'out', 'overlap']) -> np.ndarray:
    """
    * mode == 'in' (t is smaller) ::

        return[t] = [∃ i in |S| st. t ⊂ s[i]] for t in T

    * mode == 'out' (t is larger)::

         return[t] = [∃ i in |S| st. s[i] ⊂ t] for t in T

    * mode == 'overlap' ::

       return[t] = [∃ i in |S| st. s[i] ⋂ t ≠ ∅] for t in T

    returns = [min(return[T]), max(return[T])]

    :param segs: (N, 2) T-value segments
    :param t:  (R, 2) T-value segments
    :param mode:
    :return: (2, R) N-value index array
    """
    msk = _segment_overlap(segs, t, mode)
    n, t = msk.shape
    low = np.full_like(msk, n, dtype=int)
    hig = np.full_like(msk, -1, dtype=int)
    grd = np.meshgrid(np.arange(t), np.arange(n))[1]
    low[msk] = grd[msk]
    hig[msk] = grd[msk]
    low = np.min(low, axis=0)
    hig = np.max(hig, axis=0)
    low[low == n] = -1
    return np.vstack([low, hig])


def _segment_overlap(segs: Segment, t: Segment, mode: Literal['in', 'out', 'overlap']) -> np.ndarray:
    if mode == 'in':
        t1 = np.less_equal.outer(segs[:, 0], t[:, 0])  # (N, T), s[0] <= t[0]
        t2 = np.greater_equal.outer(segs[:, 1], t[:, 1])  # (N, T), s[1] >= t[1]
        return np.logical_and(t1, t2)  # (N, T)

    elif mode == 'out':
        t1 = np.greater_equal.outer(segs[:, 0], t[:, 0])  # (N, T), s[0] >= t[0]
        t2 = np.less_equal.outer(segs[:, 1], t[:, 1])  # (N, T), s[1] <= t[1]
        return np.logical_and(t1, t2)

    elif mode == 'overlap':
        t1 = np.greater_equal.outer(segs[:, 0], t[:, 1])  # (N, T), s[0] >= t[1]
        t2 = np.less_equal.outer(segs[:, 1], t[:, 0])  # (N, T), s[1] <= t[0]
        return ~np.logical_or(t1, t2)
    else:
        raise ValueError()


def segment_join(segs: SegmentLike, gap: float = 0) -> Segment:
    """

    :param segs: (N, 2) T-value segments
    :param gap: T
    :return: (N, 2) T-value segments
    """
    segs = segment_flatten(segs)
    if len(segs) <= 1:
        return segs
    ret = segs.copy()
    ret[:, 1] += gap
    ret = segment_flatten(ret)
    ret[:, 1] -= gap
    return ret


def segment_join_index(segs: Segment, gap: float = 0) -> SegmentGroup:
    """

    :param segs: (N, 2) T-value sorted segments
    :param gap: T gap
    :return: (N,) index grouping array
    """
    if segs.ndim != 2 or segs.shape[1] != 2:
        raise ValueError(f'not a (N, 2) segment array : {segs.shape}')
    elif len(segs) == 0:
        return np.array([])
    elif len(segs) == 1:
        return np.array([0])
    elif not is_sorted(segs[:, 0]):
        raise RuntimeError('segments is not sorted')

    t1 = segs[:, 0]
    t2 = segs[:, 1] + gap
    t3 = np.sign(np.diff(t2, prepend=t2[0]))
    t3[t3 < 0] = 0
    t4 = np.cumsum(np.sign(t3).astype(int))
    for t in range(1, np.max(t4) + 1):
        sx = t4 == t - 1
        s1 = np.max(t2[sx])
        s2 = t1[t4 == t][0]
        if s2 <= s1:
            t4[sx] = t
    return t4


def segment_map(f: Callable[[np.ndarray], float],
                segs: SegmentLike,
                t: np.ndarray,
                v: np.ndarray = None) -> np.ndarray:
    """

    :param f: function ((N,) V-value array) -> R-value
    :param segs: (S, 2) T-value segment
    :param t: (T,) T-value array
    :param v: (T,) V-value array. If `None`, use *t*.
    :return: (S,) R-value array
    """
    if v is None:
        v = t
    return segment_group_map(f, segment_index(as_segment(segs), t), v)


def segment_group_map(f: Callable[[np.ndarray], float],
                      group: SegmentGroup,
                      v: np.ndarray) -> np.ndarray:
    """

    :param f: function ((N,) V-value array) -> R-value
    :param group: (T) S-group array. Only non-negative value will be considered.
    :param v: (T,) T-value array
    :return: (S,) R-value array
    """
    ret = []
    for i in range(np.max(group) + 1):
        x = np.nonzero(group == i)[0]
        ret.append(f(v[x]))

    return np.array(ret)


def segment_sample(segs: SegmentLike) -> _SegmentSampleHelper:
    return _SegmentSampleHelper(segment_flatten(segs))


class _SegmentSampleHelper:
    def __init__(self, segs: Segment):
        self.segs = segs

    def random(self, time_duration: float, sample_times: int) -> Segment:
        if time_duration < 0:
            raise ValueError()
        if sample_times < 0:
            raise ValueError()
        if sample_times == 0:
            return np.zeros((0, 2), dtype=float)

        segs = self.segs[segment_duration(self.segs) >= time_duration]
        cum = np.concatenate([[0], np.cumsum(segment_duration(segs) - time_duration)])
        total = cum[-1]
        a = np.sort(np.random.random(sample_times) * total)
        i = np.searchsorted(cum, a, side='left') - 1

        ret = np.zeros((sample_times, 2), dtype=float)
        ret[:, 0] = a - cum[i] + segs[i, 0]
        ret[:, 1] = ret[:, 0] + time_duration
        return ret

    def uniform(self, time_duration: float, sample_times: int = None) -> Segment:
        if time_duration < 0:
            raise ValueError()

        if sample_times is not None:
            if sample_times < 0:
                raise ValueError()
            if sample_times == 0:
                return np.zeros((0, 2), dtype=float)

        segs = self.segs[segment_duration(self.segs) >= time_duration]
        count = (segment_duration(segs) / time_duration).astype(int)
        total = np.sum(count)
        if sample_times is None:
            sample_times = total
        if total < sample_times:
            raise RuntimeError('d*t larger than dur(segs)')

        cum = np.concatenate([[0], np.cumsum(count)])

        a = np.sort(np.random.choice(np.arange(cum[-1]), time_duration, replace=False))

        i = np.searchsorted(cum, a, side='right') - 1

        ret = np.zeros((sample_times, 2), dtype=float)
        for j in np.unique(i):
            k = np.nonzero(i == j)[0]
            n = len(k)
            assert n <= count[j], f'{n=} > {count[j]}'
            seg = segs[j]
            dur = (seg[1] - seg[0]) - time_duration * n
            assert dur >= 0
            r = np.random.random(n) * dur
            ret[k, 0] = seg[0] + np.cumsum(r) + time_duration * np.arange(n)

        ret[:, 1] = ret[:, 0] + time_duration

        return ret

    def bins(self, time_duration: float, sample_times: int = None, interval: float = 0) -> Segment:
        return segment_bins(self.segs, time_duration, interval, sample_times)


def segment_bins(segs: SegmentLike, duration: float, interval: float = 0, nbins: int = None) -> Segment:
    """
    Divide *segs* into equal-size sub-segments with equal *duration* and equal *interval*.

    ::

        returns = [(start := R[0] + (i+d)*j, start + d)] ⊆ segs, for j in [0, t)

    :param segs: (N, 2) T-value segment
    :param duration: T value
    :param interval: T value
    :param nbins: number of bins
    :return: (*nbins*, 2) T-value segment
    """
    if duration < 0:
        raise ValueError()

    if interval < 0:
        raise ValueError()

    if nbins is not None:
        if nbins < 0:
            raise ValueError()
        if nbins == 0:
            return np.zeros((0, 2), dtype=float)

    segs = segment_flatten(segs)

    # number of bins per segments
    count = ((interval + segment_duration(segs)) / (duration + interval)).astype(int)

    # maximal total bins
    total = np.sum(count)

    if nbins is None:
        nbins = total
    else:
        nbins = min(total, nbins)

    ret = np.zeros((nbins, 2), dtype=float)
    p = 0
    for j in range(len(count)):
        n = int(count[j])
        k = p + np.arange(n)
        ret[k, 0] = segs[j, 0] + (duration + interval) * np.arange(n)
        p += n

    ret[:, 1] = ret[:, 0] + duration
    return ret


# TODO shift_time