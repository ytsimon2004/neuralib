import unittest
from pathlib import Path
from typing import Final

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

from neuralib.glx import processor as p
from neuralib.glx.base import ProcessedEphysRecording
from neuralib.glx.processor import ProcessedEphysRecordingMeta
from neuralib.persistence import PersistenceHandler


class EphysProcessTester(unittest.TestCase):
    save_root: Final[Path] = Path('.test-ephys-cachr-dir')
    handler: PersistenceHandler[ProcessedEphysRecordingMeta]

    @classmethod
    def setUpClass(cls):
        from neuralib.persistence import PickleHandler
        cls.handler = PickleHandler(ProcessedEphysRecordingMeta, cls.save_root)

    @classmethod
    def tearDownClass(cls):
        import shutil
        print(f'rm -r {cls.save_root}')
        shutil.rmtree(cls.save_root)

    recording: ProcessedEphysRecording
    meta: ProcessedEphysRecordingMeta

    def setUp(self):
        data = np.random.random(size=(10, 100))  # (C, T)
        nc, nt = data.shape
        t = np.linspace(0, 10, nt)
        c = np.arange(nc)
        self.recording = ProcessedEphysRecording(t, c, data, meta=dict(test='setup'))
        self.meta = ProcessedEphysRecordingMeta('common', 'none')

        p.save(self.handler, self.meta, self.recording)

    def tearDown(self):
        meta_file = self.handler.filepath(self.meta)
        data_file = meta_file.with_suffix('.bin')
        meta_file.unlink(missing_ok=True)
        data_file.unlink(missing_ok=True)

    def test_save_meta(self):
        assert_array_equal(self.recording.channel_list, self.meta.channel_list)
        self.assertEqual(self.recording.time_start, self.meta.time_start)
        self.assertEqual(self.recording.total_samples, self.meta.total_samples)
        self.assertEqual(self.recording.sample_rate, self.meta.sample_rate)
        self.assertEqual(self.recording.dtype, self.meta.dtype)
        self.assertDictEqual(self.recording.meta, self.meta.meta)

    def test_load_meta(self):
        meta = self.handler.load_persistence(self.meta)

        self.assertIsNot(meta, self.meta)
        self.assertEqual(meta.filename, self.meta.filename)
        self.assertEqual(meta.process, self.meta.process)
        assert_array_equal(meta.channel_list, self.meta.channel_list)
        self.assertEqual(meta.time_start, self.meta.time_start)
        self.assertEqual(meta.total_samples, self.meta.total_samples)
        self.assertEqual(meta.sample_rate, self.meta.sample_rate)
        self.assertEqual(meta.dtype, self.meta.dtype)
        self.assertDictEqual(meta.meta, self.meta.meta)

    def test_load_data(self):
        recording = p.load(self.handler, self.meta)

        self.assertIsNot(self.recording, recording)
        assert_almost_equal(self.recording.t, recording.t)
        assert_almost_equal(self.recording.channel_list, recording.channel_list)
        self.assertEqual(self.recording.dtype, recording.dtype)
        assert_almost_equal(self.recording[:, :], recording[:, :])

    def test_write_ro_data(self):
        recording = p.load(self.handler, self.meta)

        with self.assertRaises(ValueError) as capture:
            recording[0, 0] = 0

        self.assertIn('assignment destination is read-only', capture.exception.args)

    def test_global_median_car(self):
        meta = p.global_car(self.handler, self.meta, method='median')
        recording = p.load(self.handler, meta)

        assert_almost_equal(self.recording.t, recording.t)
        self.assertEqual(1, len(recording.channel_list))
        assert_almost_equal(np.median(self.recording, axis=0), recording[0])

    def test_global_mean_car(self):
        meta = p.global_car(self.handler, self.meta, method='mean')
        recording = p.load(self.handler, meta)

        assert_almost_equal(self.recording.t, recording.t)
        self.assertEqual(1, len(recording.channel_list))
        assert_almost_equal(np.mean(self.recording, axis=0), recording[0])


if __name__ == '__main__':
    unittest.main()
