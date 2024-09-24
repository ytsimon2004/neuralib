import unittest

import numpy as np
from numpy.ma.testutils import assert_array_almost_equal
from numpy.testing import assert_array_equal

from neuralib.util.segments import *


class TestSegment(unittest.TestCase):
    @classmethod
    def random_segment(cls, high, length):
        return np.sort(np.random.random((2 * length,)) * high).reshape(-1, 2)

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
            self.assertTrue(np.all(ret := segment_contains(s1, i)[segment_contains(s3, i)]), f'{ret=}')
            self.assertTrue(np.all(ret := segment_contains(s2, i)[segment_contains(s3, i)]), f'{ret=}')

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
            self.assertTrue(np.all(ret := segment_contains(s3, i)[segment_contains(s1, i)]), f'{ret=}')
            self.assertTrue(np.all(ret := segment_contains(s3, i)[segment_contains(s2, i)]), f'{ret=}')

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


if __name__ == '__main__':
    unittest.main()
