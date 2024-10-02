import time
import unittest

from neuralib.io import mkdir_version, NEUROLIB_CACHE_DIRECTORY


class TestIO(unittest.TestCase):

    def test_mkdir_version(self):
        root = NEUROLIB_CACHE_DIRECTORY
        out = mkdir_version(root, 'test')
        time.sleep(3)

        self.assertTrue(out.exists())
        self.assertEqual(out, root / 'test_v0')
        out.rmdir()

    def test_mkdir_version_reuse(self):
        root = NEUROLIB_CACHE_DIRECTORY
        out1 = mkdir_version(root, 'test')
        out2 = mkdir_version(root, 'test')
        out3 = mkdir_version(root, 'test', reuse_latest=True)

        self.assertTrue(out1.exists())
        self.assertTrue(out2.exists())
        self.assertEqual(out2, out3)
        out1.rmdir()
        out2.rmdir()


if __name__ == '__main__':
    unittest.main()
