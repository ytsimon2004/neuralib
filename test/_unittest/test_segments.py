import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from neuralib.util.segments import *


# noinspection PyTypeChecker
class TestSegment(unittest.TestCase):
    @classmethod
    def random_segment(cls, high, length):
        return np.sort(np.round(np.random.random((2 * length,)) * high, 4)).reshape(-1, 2)

    def test_is_sorted(self):
        self.assertTrue(is_sorted(np.array([0, 1, 2, 3, 4])))
        self.assertFalse(is_sorted(np.array([2, 1, 2, 3, 4])))
        self.assertFalse(is_sorted(np.array([0, 1, 2, 2, 4]), strict=True))

    def test_is_sorted_random(self):
        for _ in range(100):
            a = np.random.random(10)
            b = np.sort(a)
            self.assertEqual(np.all(a == b), is_sorted(a))
            self.assertTrue(is_sorted(b))

    def test_as_segment(self):
        s = [
            [0, 2],
            [4, 6],
        ]
        assert_array_equal(np.array(s), as_segment(np.array(s)))
        assert_array_equal(np.array(s), as_segment(s))
        assert_array_equal(np.array([[0, 2]]), as_segment([0, 2]))
        assert_array_equal(np.empty((0, 2)), as_segment(np.empty((0, 2))))

    def test_as_segment_with_wrong_ndim(self):
        with self.assertRaises(ValueError):
            as_segment(np.zeros((2, 2, 2)))

    def test_as_segment_with_wrong_shape(self):
        with self.assertRaises(ValueError):
            as_segment(np.zeros((2, 3)))

    def test_segment_range(self):
        assert_array_equal(np.array([
            [0, 2],
            [4, 2],
        ]), segment_range(np.array([
            [0, 2],
            [4, 6],
        ])))

    def test_segment_duration(self):
        assert_array_equal(np.array([
            2, 2
        ]), segment_duration(np.array([
            [0, 2],
            [4, 6],
        ])))

    def test_segment_at_least_duration(self):
        for _ in range(10):
            d = np.random.randint(1, 5)

            while True:
                s = self.random_segment(100, 5)
                if np.any(segment_duration(s) < d):
                    break

            s = segment_at_least_duration(s, d)
            # segment_duration(s) may get 0.9999 for some segment, so we round them.
            self.assertTrue(np.all(np.round(segment_duration(s), 4) >= d))

    def test_segment_expand_duration(self):
        for _ in range(10):
            d = np.random.randint(1, 5)
            s = self.random_segment(100, 5)

            r = segment_expand_duration(s, d)
            assert_array_almost_equal(segment_duration(s) + d, segment_duration(r))
            assert_array_almost_equal(np.mean(r, axis=1), np.mean(s, axis=1))

    def test_segment_expand_duration_with_prepend_append(self):
        for _ in range(10):
            d = np.random.randint(1, 5)
            s = self.random_segment(100, 5)

            r = segment_expand_duration(s, (0, d))
            assert_array_almost_equal(segment_duration(s) + d, segment_duration(r))
            assert_array_almost_equal(r[:, 0], s[:, 0])
            assert_array_almost_equal(r[:, 1], s[:, 1] + d)

    def test_segment_flatten(self):
        s = [
            [0, 2],
            [4, 6],
        ]
        assert_array_equal(np.array(s), segment_flatten(s))
        assert_array_equal(np.array(s), segment_flatten(s[::-1]))
        assert_array_equal(np.array([[0, 10]]), segment_flatten([[0, 2], [2, 7], [6, 10]]))

    def test_segment_flatten_point(self):
        s = [
            [0, 0],
        ]
        assert_array_equal(np.array(s), segment_flatten(s))
        s = [
            [0, 0],
            [2, 2],
        ]
        assert_array_equal(np.array(s), segment_flatten(s))

    def test_segment_flatten_consume_inf(self):
        assert_array_equal(np.array([
            [-np.inf, 6],
        ]), segment_flatten([
            [-np.inf, -np.inf],
            [-np.inf, 6],
        ]))
        assert_array_equal(np.array([
            [10, np.inf],
        ]), segment_flatten([
            [10, np.inf],
            [np.inf, np.inf],
        ]))

    def test_segment_flatten_inf(self):
        assert_array_equal(np.array([
            [-np.inf, np.inf],
        ]), segment_flatten([
            [-np.inf, 0],
            [0, np.inf],
        ]))
        assert_array_equal(np.array([
            [-np.inf, 0],
            [10, np.inf],
        ]), segment_flatten([
            [-np.inf, 0],
            [10, np.inf],
        ]))

    def test_segment_flatten_opened(self):
        assert_array_equal(np.array([
            [0, 2],
            [2, 4],
        ]), segment_flatten([
            [0, 2],
            [2, 4],
        ], closed=False))
        assert_array_equal(np.array([
            [0, 4],
        ]), segment_flatten([
            [0, 2],
            [2, 4],
        ], closed=True))

    def test_segment_flatten_opened_case_1(self):
        s = [
            [2.5, 2.7],
            [2.7, 2.9],
            [2.9, 3.1],
            [3.9, 4.1],
            [4.1, 4.3],
            [4.3, 4.5],
            [4.5, 4.7],
            [4.7, 4.9],
            [6.1, 6.3],
            [6.3, 6.5]
        ]
        assert_array_equal(np.array(s), segment_flatten(s, closed=False))

    def test_segment_invert(self):
        s = [
            [0, 2],
            [4, 6],
        ]
        r = [
            [-np.inf, 0],
            [2, 4],
            [6, np.inf]
        ]
        assert_array_equal(np.array(r), segment_invert(s))
        assert_array_equal(np.array(s), segment_invert(r))

    def test_segment_invert_half_side_inf(self):
        s = [
            [0, 2],
            [4, np.inf],
        ]
        r = [
            [-np.inf, 0],
            [2, 4],
        ]
        assert_array_equal(np.array(r), segment_invert(s))
        assert_array_equal(np.array(s), segment_invert(r))

    def test_segment_invert_on_point(self):
        s = [
            [0, 0]
        ]
        r = [
            [-np.inf, 0],
            [0, np.inf],
        ]
        assert_array_equal(np.array(r), segment_invert(s))
        assert_array_equal(np.array(s), segment_invert(r))

    def test_segment_invert_empty_set(self):
        e = np.empty((0, 2))
        u = segment_universe()
        assert_array_equal(u, segment_invert(e))
        assert_array_equal(e, segment_invert(u))

    def test_segment_intersection(self):
        for _ in range(10):
            s1 = self.random_segment(20, 5)
            s2 = self.random_segment(20, 5)
            s3 = segment_intersection(s1, s2)
            assert_array_equal(s3, segment_intersection(s1, s3))
            assert_array_equal(s3, segment_intersection(s2, s3))

    def test_segment_intersection_contain(self):
        i = np.arange(20)
        for _ in range(10):
            s1 = self.random_segment(20, 5)
            s2 = self.random_segment(20, 5)
            s3 = segment_intersection(s1, s2)

            c1 = segment_contains(s1, i)
            c2 = segment_contains(s2, i)
            c3 = segment_contains(s3, i)
            assert_array_equal(c3, np.logical_and(c1, c2))

    def test_segment_intersection_on_empty_set(self):
        e = np.empty((0, 2))
        for _ in range(10):
            s = self.random_segment(20, 5)
            assert_array_equal(e, segment_intersection(s, e))
            assert_array_equal(e, segment_intersection(e, s))
            assert_array_equal(e, segment_intersection(e, e))

    def test_segment_intersection_on_disjoint_set(self):
        e = np.empty((0, 2))
        for _ in range(10):
            s = self.random_segment(20, 5)
            assert_array_equal(e, segment_intersection(s[0:2], s[-2:]))

    def test_segment_union(self):
        for _ in range(10):
            s1 = self.random_segment(20, 5)
            s2 = self.random_segment(20, 5)
            s3 = segment_union(s1, s2)
            assert_array_equal(s3, segment_union(s1, s3))
            assert_array_equal(s3, segment_union(s2, s3))

    def test_segment_union_contain(self):
        i = np.arange(20)
        for _ in range(10):
            s1 = self.random_segment(20, 5)
            s2 = self.random_segment(20, 5)
            s3 = segment_union(s1, s2)

            c1 = segment_contains(s1, i)
            c2 = segment_contains(s2, i)
            c3 = segment_contains(s3, i)
            assert_array_equal(c3, np.logical_or(c1, c2))

    def test_segment_union_on_empty_set(self):
        e = np.empty((0, 2))
        for _ in range(10):
            s = self.random_segment(20, 5)
            assert_array_equal(s, segment_union(s, e))
            assert_array_equal(s, segment_union(e, s))
            assert_array_equal(e, segment_union(e, e))

    def test_segment_union_on_disjoint_set(self):
        for _ in range(10):
            s = self.random_segment(20, 5)
            assert_array_equal(segment_flatten(s), segment_union(s[0:5], s[5:]))

    def test_segment_diff(self):
        e = np.empty((0, 2))

        for _ in range(10):
            s1 = self.random_segment(20, 5)
            s2 = self.random_segment(20, 5)
            s3 = segment_intersection(s1, s2)
            d12 = segment_diff(s1, s2)
            d21 = segment_diff(s2, s1)

            assert_array_equal(segment_flatten(s1), segment_union(s3, d12))
            assert_array_equal(segment_flatten(s2), segment_union(s3, d21))
            assert_array_equal(e, segment_intersection(s2, d12))
            assert_array_equal(e, segment_intersection(s1, d21))
            assert_array_equal(d12.ravel(), segment_diff(s1, s3).ravel())
            assert_array_equal(d21.ravel(), segment_diff(s2, s3).ravel())

    def test_segment_diff_on_empty_set(self):
        e = np.empty((0, 2))
        for _ in range(10):
            s = self.random_segment(20, 5)
            assert_array_equal(s, segment_diff(s, e))
            assert_array_equal(e, segment_diff(e, s))
            assert_array_equal(e, segment_diff(e, e))

    def test_segment_diff_on_disjoint_set(self):
        for _ in range(10):
            while True:
                s = self.random_segment(20, 5)
                if len(s) > 4:
                    break
            assert_array_equal(s[0:2], segment_diff(s[0:2], s[-2:]))

    def test_segment_contains(self):
        i = np.arange(20)
        for _ in range(10):
            s = self.random_segment(20, 5)
            c = segment_contains(s, i)
            for ii in i:
                self.assertEqual(c[ii], np.any(np.logical_and(
                    s[:, 0] <= ii,
                    ii <= s[:, 1]
                )))

    def test_segment_index(self):
        t = np.arange(-5, 25)
        for _ in range(10):
            s = self.random_segment(20, 5)
            i = segment_index(s, t)

            for j in np.nonzero(i >= 0)[0]:
                self.assertTrue(s[i[j], 0] <= t[j] <= s[i[j], 1])

            for j in np.nonzero(i < 0)[0]:
                if -i[j] - 1 == 0:
                    self.assertTrue(t[j] < s[0, 0])
                elif -i[j] - 1 == len(s):
                    self.assertTrue(s[-1, 1] <= t[j])
                else:
                    self.assertTrue(s[-i[j] - 2, 1] < t[j] < s[-i[j] - 1, 0])

    def test_segment_overlap_mode_in(self):
        for _ in range(10):
            while True:
                s1 = self.random_segment(20, 5)
                s2 = self.random_segment(20, 5)
                if len(s3 := segment_intersection(s1, s2)) > 0:
                    break

            self.assertTrue(np.all(segment_overlap(s1, s1, mode='in')))
            self.assertTrue(np.all(segment_overlap(s1, s3, mode='in')))
            self.assertTrue(np.all(segment_overlap(s2, s3, mode='in')))

            self.assertFalse(np.all(segment_overlap(s3, s1, mode='in')))
            self.assertFalse(np.all(segment_overlap(s3, s2, mode='in')))

    def test_segment_overlap_mode_out(self):
        for _ in range(10):
            s1 = self.random_segment(20, 5)
            s2 = segment_expand_duration(s1, 0.1)

            self.assertTrue(np.all(segment_overlap(s1, s1, mode='out')))
            self.assertTrue(np.all(segment_overlap(s1, s2, mode='out')))
            self.assertTrue(np.all(segment_overlap(s2, s1, mode='in')))

    def test_segment_overlap_mode_overlap(self):
        for _ in range(10):
            s1 = segment_at_least_duration(self.random_segment(20, 5), 0.2)
            s2 = s1 + 0.1

            self.assertTrue(np.all(segment_overlap(s1, s2, mode='overlap')))
            self.assertTrue(np.all(segment_overlap(s2, s1, mode='overlap')))

            self.assertFalse(np.all(segment_overlap(s1, s2, mode='in')))
            self.assertFalse(np.all(segment_overlap(s1, s2, mode='out')))

    def test_segment_overlap_index_mode_in(self):
        for _ in range(10):
            while True:
                s1 = self.random_segment(20, 5)
                s2 = self.random_segment(20, 5)
                if len(s3 := segment_intersection(s1, s2)) > 0:
                    break

            i = segment_overlap_index(s1, s2, mode='in')
            for j, (a, b) in enumerate(i.T):
                if a < 0:
                    self.assertTrue(b < 0)
                else:
                    self.assertTrue(s1[a, 0] <= s2[j, 0])
                    self.assertTrue(s2[j, 1] <= s1[b, 1])
                    self.assertTrue(a == 0 or s1[a - 1, 1] < s2[j, 0])
                    self.assertTrue(b + 1 == len(s1) or s2[j, 1] < s1[b + 1, 0])

    def test_segment_overlap_index_mode_out(self):
        for _ in range(10):
            s1 = self.random_segment(20, 5)
            s2 = self.random_segment(20, 5)
            s3 = segment_union(s1, s2)

            i = segment_overlap_index(s1, s3, mode='out')
            for j, (a, b) in enumerate(i.T):
                if a < 0:
                    self.assertTrue(b < 0)
                else:
                    self.assertTrue(s3[j, 0] <= s1[a, 0])
                    self.assertTrue(s1[b, 1] <= s3[j, 1])

    def test_segment_overlap_index_mode_overlap(self):
        for _ in range(10):
            while True:
                s1 = self.random_segment(20, 5)
                s2 = self.random_segment(20, 5)
                if len(s3 := segment_intersection(s1, s2)) > 0:
                    break

            i = segment_overlap_index(s1, s2, mode='overlap')
            for j, (a, b) in enumerate(i.T):
                if a < 0:
                    self.assertTrue(b < 0)
                else:
                    self.assertTrue(s1[a, 0] <= s2[j, 0] or s2[j, 0] <= s1[a, 0])
                    self.assertTrue(s2[j, 1] <= s1[b, 1] or s1[b, 1] <= s2[j, 1])
                    self.assertTrue(a == 0 or s1[a - 1, 1] < s2[j, 0])
                    self.assertTrue(b + 1 >= len(s1) or s2[j, 1] < s1[b + 1, 0])

    def test_segment_sample_random(self):
        for _ in range(10):
            s1 = self.random_segment(20, 5)
            s2 = segment_sample(s1).random(0.1, 10)

            self.assertTrue(np.all(np.round(segment_duration(s2), 4) == 0.1))
            self.assertTrue(np.all(segment_overlap(s1, s2, mode='in')))

    def test_segment_sample_uniform(self):
        for _ in range(10):
            s1 = self.random_segment(20, 5)
            s2 = segment_sample(s1).uniform(0.1, 10)

            self.assertTrue(np.all(np.round(segment_duration(s2), 4) == 0.1))
            self.assertTrue(np.all(ret := segment_overlap(s1, s2, mode='in')), f'{ret}')
            assert_array_equal(s2, segment_flatten(s2))

    def test_segment_sample_bins(self):
        for _ in range(10):
            # we need to round to 1, because few segment may be union-ed due to floating number precise when segment_flatten
            # we tested segment_flatten for precise 1 case in test_segment_flatten_opened_case_1
            # to ensure it works in general.
            s1 = np.round(self.random_segment(20, 5), 1)
            s2 = np.round(segment_sample(s1).bins(0.2, 10), 1)

            self.assertTrue(np.all(np.round(segment_duration(s2), 4) == 0.2))
            self.assertTrue(np.all(ret := segment_overlap(s1, s2, mode='in')), f'{ret}')
            assert_array_equal(s2, ret := segment_flatten(s2, closed=False), f'{s2} != {ret}')

            o = segment_overlap_index(s1, s2, mode='in')[0]
            for oo in np.unique(o):
                if np.count_nonzero(o == oo) > 1:
                    self.assertAlmostEqual(0.2, np.min(np.diff(s2[o == oo, 0])))

    def test_segment_sample_bins_with_interval(self):
        for _ in range(10):
            s1 = self.random_segment(20, 5)
            s2 = segment_sample(s1).bins(0.2, interval=0.1)

            self.assertTrue(np.all(np.round(segment_duration(s2), 4) == 0.2))
            self.assertTrue(np.all(ret := segment_overlap(s1, s2, mode='in')), f'{ret}')
            assert_array_equal(s2, segment_flatten(s2, closed=False))

            o = segment_overlap_index(s1, s2, mode='in')[0]
            for oo in np.unique(o):
                if np.count_nonzero(o == oo) > 1:
                    self.assertAlmostEqual(0.3, np.min(np.diff(s2[o == oo, 0])))

    def test_shift_time(self):
        t = np.arange(10)
        s = np.array([[3, 7]])
        e = np.array(
            [0, 1, 2, 3.2, 4.2, 5.2, 6.2, 7, 8, 9], dtype=float
        )
        a = shift_time(t, 0.2, s)
        assert_array_almost_equal(e, a)

    def test_shift_time_circular(self):
        t = np.arange(10)
        s = np.array([[3, 7.1]])
        e = np.array(
            [0, 1, 2, 3.1, 3.2, 4.2, 5.2, 6.2, 8, 9], dtype=float
        )
        a = shift_time(t, 0.2, s, circular=True)
        assert_array_almost_equal(e, a)

    def test_shift_time_random(self):
        t = np.arange(20)
        for _ in range(10):
            s = self.random_segment(20, 5)
            a = shift_time(t, 0.2, s)
            self.assertGreaterEqual(len(t), len(a))
            assert_array_equal(t[~segment_contains(s, t)], a[~segment_contains(s, a)])
            with self.assertRaises(AssertionError):
                assert_array_equal(t[segment_contains(s, t)], a[segment_contains(s, a)])

    def test_shift_time_circular_random(self):
        t = np.arange(20)
        for _ in range(10):
            s = self.random_segment(20, 5)
            a = shift_time(t, 0.2, s, circular=True)
            self.assertEqual(len(t), len(a))
            assert_array_equal(t[~segment_contains(s, t)], a[~segment_contains(s, a)])
            with self.assertRaises(AssertionError):
                assert_array_equal(t[segment_contains(s, t)], a[segment_contains(s, a)])

    def test_shuffle_time_uniform(self):
        t = np.arange(20)
        for _ in range(10):
            s = self.random_segment(20, 5)
            a = shuffle_time_uniform(t, s)
            self.assertGreaterEqual(len(t), len(a))
            assert_array_equal(t[~segment_contains(s, t)], a[~segment_contains(s, a)])
            with self.assertRaises(AssertionError):
                assert_array_equal(t[segment_contains(s, t)], a[segment_contains(s, a)])

    def test_shuffle_time_uniform_circular(self):
        t = np.arange(20)
        for _ in range(10):
            s = self.random_segment(20, 5)
            a = shuffle_time_uniform(t, s, circular=True)
            self.assertEqual(len(t), len(a))
            assert_array_equal(t[~segment_contains(s, t)], a[~segment_contains(s, a)])
            with self.assertRaises(AssertionError):
                assert_array_equal(t[segment_contains(s, t)], a[segment_contains(s, a)])

    def test_shuffle_time_normal(self):
        t = np.arange(20)
        for _ in range(10):
            s = self.random_segment(20, 5)
            a = shuffle_time_normal(t, 0, 10, s)
            self.assertGreaterEqual(len(t), len(a))
            assert_array_equal(t[~segment_contains(s, t)], a[~segment_contains(s, a)])
            with self.assertRaises(AssertionError):
                assert_array_equal(t[segment_contains(s, t)], a[segment_contains(s, a)])

    def test_shuffle_time_normal_circular(self):
        t = np.arange(20)
        for _ in range(10):
            s = self.random_segment(20, 5)
            a = shuffle_time_normal(t, 0, 10, s, circular=True)
            self.assertEqual(len(t), len(a))
            assert_array_equal(t[~segment_contains(s, t)], a[~segment_contains(s, a)])
            with self.assertRaises(AssertionError):
                assert_array_equal(t[segment_contains(s, t)], a[segment_contains(s, a)])


if __name__ == '__main__':
    unittest.main()
