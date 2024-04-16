import unittest

import numpy as np
from numpy.testing import assert_array_equal

from neuralib.glx.allocator import *


class TestGlxAllocator(unittest.TestCase):
    def test_default_with_shape(self):
        allocator = default_allocator()
        assert_array_equal(allocator((2, 2), int), np.zeros((2, 2), int), strict=True)
        assert_array_equal(allocator((2, 2), float), np.zeros((2, 2), float), strict=True)
        assert_array_equal(allocator((0, 2), float), np.zeros((0, 2), float), strict=True)

    def test_default_with_array(self):
        allocator = default_allocator()

        x = np.zeros((2, 2), int)
        assert_array_equal(allocator(x), x, strict=True)
        assert_array_equal(allocator(x, float), np.zeros((2, 2), float), strict=True)

        with self.assertRaises(AssertionError):
            assert_array_equal(allocator(x, float), x, strict=True)

    def test_empty_with_shape(self):
        allocator = empty_allocator()

        x = np.zeros((2, 2), int)
        a = allocator((2, 2))
        self.assertEqual(a.shape, x.shape)

    def test_empty_with_array(self):
        allocator = empty_allocator()

        x = np.zeros((2, 2), int)
        a = allocator(x)
        self.assertEqual(a.shape, x.shape)


if __name__ == '__main__':
    unittest.main()
