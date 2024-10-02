import sys
import unittest
from pathlib import Path
from typing import Final

import numpy as np

from neuralib.ephys.glx.spikeglx import GlxMeta, GlxRecording, GlxIndex
from neuralib.io.dataset import _load_ephys_meta, load_ephys_meta, load_ephys_data


class TestGlxMeta(unittest.TestCase):
    context = None
    meta: GlxMeta

    @classmethod
    def setUpClass(cls):
        cls.context = load_ephys_meta()
        cls.meta = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.meta = None
        cls.context.__exit__()

    def test_load_meta(self):
        meta = self.meta
        self.assertEqual(385, meta.total_channels)
        self.assertEqual(30648, meta.total_samples)
        self.assertEqual(30000, meta.sample_rate)
        self.assertEqual(1.0216, meta.total_duration)

    @unittest.skipUnless(sys.version_info >= (3, 10), 'python version below 3.10')
    def test_load_channel_info(self):
        _ = self.meta.channel_info()

    @unittest.skipIf(sys.version_info >= (3, 10), 'python version above 3.9')
    def test_load_channel_info_under_py310(self):
        with self.assertRaises(AttributeError):
            self.meta.channel_info()


class TestGlxRecording(unittest.TestCase):
    context = None
    data: GlxRecording

    @classmethod
    def setUpClass(cls):
        cls.context = load_ephys_data()
        cls.data = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.data = None
        cls.context.__exit__()

    def test_load_data(self):
        data = self.data
        self.assertEqual(385, data.total_channels)
        self.assertEqual(30648, data.total_samples)
        self.assertEqual(30000, data.sample_rate)
        self.assertEqual(1, data.total_duration)

    def test_data_read(self):
        self.assertEqual(385, len(self.data[:, 0]))
        self.assertEqual(30648, len(self.data[0, :]))

    def test_data_read_out_of_bound(self):
        with self.assertRaises(IndexError):
            self.data[388, 0]

    @unittest.skipUnless(sys.version_info >= (3, 10), 'python version below 3.10')
    def test_load_channel_info(self):
        _ = self.data.channel_info()

    @unittest.skipIf(sys.version_info >= (3, 10), 'python version above 3.9')
    def test_load_channel_info_under_py310(self):
        with self.assertRaises(NotImplementedError):
            self.data.channel_info()


class TestGlxIndex(unittest.TestCase):
    def test_parse_filename(self):
        self.assertTupleEqual(GlxIndex('example', 0, '0', 0),
                              GlxIndex.parse_filename('example_g0_t0.imec0.ap.bin'))
        self.assertTupleEqual(GlxIndex('long_example_name', 1, '0', 1),
                              GlxIndex.parse_filename('long_example_name_g1_t0.imec1.ap.bin'))
        self.assertTupleEqual(GlxIndex('catgt_example', 1, 'cat', 0),
                              GlxIndex.parse_filename('catgt_example_g1_tcat.imec0.ap.bin'))
        self.assertTupleEqual(GlxIndex('catgt_example', 1, 'super', 0),
                              GlxIndex.parse_filename('catgt_example_g1_tsuper.imec0.ap.bin'))

    def test_as_cat_index(self):
        index = GlxIndex('example', 0, '0', 0)
        self.assertFalse(index.is_catgt)
        self.assertFalse(index.is_supercat)

        self.assertTupleEqual(GlxIndex('example', 0, 'cat', 0), ret := index.as_cat_index())
        self.assertTrue(ret.is_catgt)
        self.assertFalse(ret.is_supercat)

        self.assertTupleEqual(GlxIndex('example', 0, 'cat', 0), index.as_cat_index().as_cat_index())

    def test_as_super_index(self):
        index = GlxIndex('example', 0, '0', 0)
        self.assertFalse(index.is_catgt)
        self.assertFalse(index.is_supercat)

        self.assertTupleEqual(GlxIndex('example', 0, 'super', 0), index.as_super_index())
        self.assertTupleEqual(GlxIndex('example', 0, 'super', 0), index.as_cat_index().as_super_index())
        self.assertTupleEqual(GlxIndex('example', 0, 'super', 0), ret := index.as_super_index().as_super_index())
        self.assertTrue(ret.is_catgt)
        self.assertTrue(ret.is_supercat)

    @unittest.skip('directory name is SpikeGLX version depend')
    def test_dirname(self):
        index = GlxIndex('example', 0, '0', 0)
        self.assertEqual('', index.dirname())

    def test_filename(self):
        filename = 'example_g0_t0.imec0.ap.bin'
        index = GlxIndex.parse_filename('example_g0_t0.imec0.ap.bin')
        self.assertEqual(filename, index.filename())
        self.assertEqual('example_g0_t0.imec0.lf.bin', index.filename(f='lf'))
        self.assertEqual('example_g0_t0.imec0.ap.meta', index.filename(ext='.meta'))


if __name__ == '__main__':
    unittest.main()
