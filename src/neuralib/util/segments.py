from collections.abc import Callable
from typing import Literal, Union, Iterable

import numpy as np
from neuralib.typing import ArrayLike

__all__ = [
    'SegmentLike',
    'segment_bool_mask',
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
    'shuffle_time',
    'shuffle_time_uniform',
    'shuffle_time_normal',
    'shift_time',
    'foreach_map',
    'grouped_iter'
]

Segment = np.ndarray  # (N, 2) value array ([(start, stop)]), as a segment.
SegmentLike = Union[Segment, tuple[float, float], list[tuple[float, float]]]
SegmentGroup = np.ndarray  # (N,) int array, where a unique value indicate a unique group/segment.


def segment_bool_mask(mx: np.ndarray) -> Segment:
    """
    :param mx: `Array[bool, N]`
    :return: ``Segment``. `Array[int, [N, 2]]`
    """
    if mx.ndim != 1:
        raise ValueError('not a (N, 1) segment')

    if mx.dtype != np.bool_:
        raise TypeError('not a bool array')

    diff = np.diff(mx.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    return np.column_stack((starts, ends))


def is_sorted(a: np.ndarray, strict: bool = False) -> bool:
    """
    Check if array is sorted.
    [reference](https://stackoverflow.com/a/47004507)

    :param a: Input array to check for sorted order.
    :param strict: If True, checks for strictly increasing order. Otherwise, checks for non-decreasing order.
    :return: Returns True if the input array is sorted based on the specified criteria, else False.
    """
    if strict:
        return np.all(a[:-1] < a[1:])
    else:
        return np.all(a[:-1] <= a[1:])


def segment_mask(x: np.ndarray,
                 t: np.ndarray | None = None,
                 duration: float | None = None,
                 merge: float | None = None) -> SegmentGroup:
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


def segment_epochs(x: np.ndarray,
                   t: np.ndarray | None = None,
                   duration: float | None = None,
                   merge: float | None = None) -> Segment:
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
    """
    :param segs: Input segment-like data that can be converted to a 2D array of floats.
    :return: A numpy 2D array representation of the input segments.
    :raises ValueError: If the input cannot be reshaped to a (N, 2) segment array.
    """
    ret = np.atleast_2d(segs).astype(float, copy=False)
    if ret.ndim != 2 or ret.shape[1] != 2:
        raise ValueError(f'not a (N, 2) segment array : {ret.shape}')
    return ret


def segment_range(segs: SegmentLike) -> np.ndarray:
    """change [(start, stop)] to [(start, duration)]

    :param segs: segments. `Array[float, [N, 2]]`
    :return: `Array[int, [N, 2]]`
    """
    a = as_segment(segs)
    r = a.copy()
    r[:, 1] = a[:, 1] - a[:, 0]
    return r


def segment_duration(segs: SegmentLike) -> np.ndarray:
    """

    :param segs: T-value segments. `Array[float, [N, 2]]`
    :return: T-value array. `Array[int, N]`
    """
    a = as_segment(segs)
    return a[:, 1] - a[:, 0]


def segment_at_least_duration(segs: SegmentLike, duration: float) -> np.ndarray:
    """

    :param segs: T-value segments. `Array[float, [N, 2]]`
    :param duration: T-value
    :return: T-value segments. `Array[float, [N, 2]]`
    """
    a = as_segment(segs).copy()
    dur = duration - (a[:, 1] - a[:, 0])
    ext = np.where(dur > 0, dur / 2, 0)
    a[:, 0] -= ext
    a[:, 1] += ext

    return a


def segment_expand_duration(segs: SegmentLike, duration: float | tuple[float, float] | np.ndarray) -> np.ndarray:
    """

    :param segs: T-value segments. `Array[float, [N, 2]]`
    :param duration: T-value scalar, tuple(prepend:T, append:T), or (N, 2?) array
    :return: T-value segments. `Array[float, [N, 2]]`
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
    return np.array([[-np.inf, np.inf]])


def segment_flatten(segs: SegmentLike, closed: bool = True) -> Segment:
    """sorting and remove overlapped segments.

    :param segs: T-value segments. `Array[float, [N, 2]]`
    :param closed: Is segment a closed on right side?
    :return: T-value segments. `Array[float, [M, 2]]`
    """
    a = as_segment(segs)

    if len(a) == 0:
        return a

    if not is_sorted(a[:, 0]):
        a = a[np.argsort(a[:, 0])]

    if is_sorted(a.ravel(), strict=True):
        return a

    r = np.empty_like(a)
    r[0] = a[0]
    i = 1
    j = 1
    while j < len(a):
        p: float = r[i - 1, 1]
        q: float = a[j, 0]
        assert r[i - 1, 0] <= q, f'{r[i-1]=} <> {a[j]=}'

        if (p >= q if closed else p > q):
            r[i - 1, 1] = max(a[j, 1], p)
        else:
            r[i] = a[j]
            i += 1
        j += 1

    r = r[:i]
    return r


def segment_invert(segs: SegmentLike) -> Segment:
    """

    :param segs: T-value segments. `Array[float, [N, 2]]`
    :return:  T-value segments. `Array[float, [N+1, 2]]`
    """
    return _segment_invert(segment_flatten(segs, closed=False))


def _segment_invert(segs: Segment) -> Segment:
    if len(segs) == 0:
        return segment_universe()

    a = np.concatenate([[-np.inf], segs.ravel(), [np.inf]]).reshape((-1, 2))

    d = []
    if np.all(np.isinf(a[0])):
        d.append(0)
    if np.all(np.isinf(a[-1])):
        d.append(-1)

    return np.delete(a, d, axis=0)


def segment_intersection(a: SegmentLike, b: SegmentLike) -> Segment:
    """

    :param a: T-value segments. `Array[float, [A, 2]]`
    :param b: T-value segments. `Array[float, [B, 2]]`
    :return: T-value segments. `Array[float, [C, 2]]`
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
            if s < q:  # p <= s < {q,t}
                ret.append((s, min(q, t)))
        else:
            if p < t:  # s < p < {q,t}
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

    :param a: T-value segments. `Array[float, [A, 2]]`
    :param b: T-value segments. `Array[float, [B, 2]]`
    :param gap: T value
    :return: T-value segments. `Array[float, [C, 2]]`
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

    :param a: T-value segments. `Array[float, [A, 2]]`
    :param b: T-value segments. `Array[float, [B, 2]]`
    :return: T-value segments. `Array[float, [C, 2]]`
    """
    a = segment_flatten(a)
    b = segment_flatten(b)
    return segment_intersection(a, _segment_invert(b))


def segment_contains(segs: SegmentLike, t: np.ndarray) -> np.ndarray:
    """
    whether *t* in the segments.

    :param segs: T-value segments. `Array[float, [N, 2]]`
    :param t: T-value array. `Array[float, R]`
    :return: bool array. `Array[bool, R]`
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

        return[t] = [∃ i in |S| st. t ⊂ s[i]] for t in T, otherwise [-1]

    * mode == 'out' (t is larger)::

         return[t] = [∃ i in |S| st. s[i] ⊂ t] for t in T, otherwise [-1]

    * mode == 'overlap' ::

       return[t] = [∃ i in |S| st. s[i] ⋂ t ≠ ∅] for t in T, otherwise [-1]

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


def segment_sample(segs: SegmentLike) -> '_SegmentSampleHelper':
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

        a = np.sort(np.random.choice(np.arange(cum[-1]), sample_times, replace=False))

        i = np.searchsorted(cum, a, side='right') - 1

        ret = np.zeros((sample_times, 2), dtype=float)
        for j in np.unique(i):
            k = np.nonzero(i == j)[0]
            n = len(k)
            assert n <= count[j], f'{n=} > {count[j]}'
            seg = segs[j]
            dur = (seg[1] - seg[0]) - time_duration * n
            assert dur >= 0
            r = np.sort(np.random.random(n)) * dur
            ret[k, 0] = seg[0] + r + time_duration * np.arange(n)

        ret[:, 1] = ret[:, 0] + time_duration

        return ret

    def bins(self, time_duration: float, sample_times: int = None, interval: float = 0) -> Segment:
        return segment_bins(self.segs, time_duration, interval, sample_times)


def segment_bins(segs: SegmentLike,
                 duration: float,
                 interval: float = 0,
                 nbins: int = None) -> Segment:
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
        n = min(int(count[j]), nbins - p)
        k = p + np.arange(n)
        ret[k, 0] = segs[j, 0] + (duration + interval) * np.arange(n)
        p += n
        if p >= nbins:
            break

    ret[:, 1] = ret[:, 0] + duration
    return ret


def shuffle_time(t: np.ndarray,
                 method: Callable[[np.ndarray], np.ndarray],
                 segs: Segment = None,
                 duration: float = np.inf,
                 circular=True):
    """
    Shuffle *t* by remapping function *method* for *t* in *segs*.

    :param t: (N,) T-value array
    :param method: time remapping function with signature ((K,) T-value array) -> (K,) T-value array
    :param segs: (S, 2) segment, only shift time in the segments.
    :param duration: The maximal T value
    :param circular: keep total event number. If epochs is given, keep total event number in epochs.
    :return: (N',) T-value array
    """
    if segs is None:
        ret = method(t)

        if not np.isinf(duration):
            if circular:
                while np.any(x := ret > duration):
                    ret[x] -= duration
            else:
                ret = np.delete(ret, ret > duration, 0)
    else:
        ret = _shuffle_time_in_segment(t, method, segs, (0, duration), circular=circular)

    return np.sort(ret)


def shuffle_time_uniform(t: np.ndarray,
                         segs: Segment = None,
                         *,
                         duration: float = np.inf,
                         circular=True) -> np.ndarray:
    """
    Shuffle *t* for *t* in *segs*.

    :param t: (N,) T-value array
    :param segs: (S, 2) segment, only shift time in the segments.
    :param duration: The maximal T value
    :param circular: keep total event number. If epochs is given, keep total event number in epochs.
    :return: (N',) T-value array
    """
    if np.isinf(duration):
        duration = np.max(t)

    def method(it: np.ndarray) -> np.ndarray:
        return np.random.uniform(0, duration, size=it.shape)

    return shuffle_time(t, method, segs, duration, circular)


def shuffle_time_normal(t: np.ndarray,
                        loc: float = 0,
                        scale: float = 1,
                        segs: Segment = None,
                        *,
                        duration: float = np.inf,
                        circular=True) -> np.ndarray:
    """
    Shuffle *t* with add a value from a normal distribution for *t* in *segs*.

    :param t: (N,) T-value array
    :param loc: mean of the normal distribution
    :param scale: std of the normal distribution
    :param segs: (S, 2) segment, only shift time in the segments.
    :param duration: The maximal T value
    :param circular: keep total event number. If epochs is given, keep total event number in epochs.
    :return: (N',) T-value array
    """

    def method(it: np.ndarray) -> np.ndarray:
        return it + np.random.normal(loc, scale, size=it.shape)

    return shuffle_time(t, method, segs, duration, circular)


def shift_time(t: np.ndarray,
               shift: float,
               segs: Segment = None,
               *,
               duration: float = np.inf,
               circular=True):
    """
    Shift *t* with a *shift* value for *t* in *segs*.

    :param t: (N,) T-value array
    :param shift: shift T value, positive value.
    :param segs:  (S, 2) segment, only shift time in the segments.
    :param duration: The maximal T value for wrapping when *circular*
    :param circular: keep total event number. If epochs is given, keep total event number in epochs.
    :return: (N',) T-value array
    """
    if circular:
        if not (0 <= shift <= duration):
            raise ValueError(f'illegal {shift=}')

    def method(it: np.ndarray) -> np.ndarray:
        return it + shift

    return shuffle_time(t, method, segs, duration, circular)


def _shuffle_time_in_segment(t: np.ndarray,
                             shift: Callable[[np.ndarray], np.ndarray],
                             epochs: Segment,
                             duration: tuple[float, float],
                             circular=True):
    epochs = segment_flatten(epochs)  # Array[float, E', 2]

    ret = t.astype(float, copy=True)  # Array[float, T]
    i = segment_index(epochs, t)  # Array[E', T], index of seg for all t
    x = np.nonzero(i >= 0)[0]  # Array[T, T'], index of t for valid t
    i = i[x]  # Array[E', T'], index of seg for valid t
    sd = segment_duration(epochs)  # Array[float, E']
    sd = np.concatenate([[0], np.cumsum(sd)])  # Array[float, E' + 1]
    ret[x] = shift(ret[x] - epochs[i, 0] + sd[i])

    if circular:
        ret[x] %= sd[-1]
    else:
        o = x[ret[x] > duration[1]]
        ret[o] = np.nan
        o = x[ret[x] < duration[0]]
        ret[o] = np.nan
        x = x[~np.isnan(ret[x])]

    i = np.searchsorted(sd, ret[x], side='right') - 1  # Array[E', T'], new index of seg for shifted t
    ret[x] = ret[x] + epochs[i, 0] - sd[i]

    if not circular:
        ret = ret[~np.isnan(ret)]

    return ret


def foreach_map(v: np.ndarray,
                f: Callable[[np.ndarray], np.ndarray],
                indices_or_sections: np.ndarray) -> np.ndarray:
    """
    Map function ``f`` into vector ``v`` in given sections.

    :param v: Input 1d array
    :param f: Function to apply to each segment
    :param indices_or_sections: If indices_or_sections is an integer, N, the array will be divided into N equal arrays along axis.
        If such a split is not possible, an error is raised
    :return:
    """
    if v.ndim != 1 or indices_or_sections.ndim != 1:
        raise RuntimeError('')

    split_list = np.split(v, indices_or_sections)

    return np.array(list(map(f, split_list[:-1])))  # avoid empty list if divisible


def grouped_iter(v: ArrayLike | Iterable, n: int) -> zip:
    """
    Groups elements from the input iterable ``v`` into tuples of length ``n``

    >>> list(grouped_iter([1, 2, 3, 4, 5, 6], 2))
    [(1, 2), (3, 4), (5, 6)]

    :param v: input iterable to be grouped.
    :param n: number of elements per group
    :return: An iterator over tuples of length n
    """
    return zip(*[iter(v)] * n)
