import unittest

import numpy as np


@unittest.skip('run manually, since module "allensdk" build error')
class TestAtlas(unittest.TestCase):

    # ========== #
    # Atlas View #
    # ========== #

    def test_load_ccf_annotation(self):
        from neuralib.atlas.data import load_ccf_annotation
        arr = load_ccf_annotation()

        self.assertEquals(arr.dtype, np.uint16)
        self.assertTupleEqual(arr.shape, (1320, 800, 1140))

    def test_load_ccf_template(self):
        from neuralib.atlas.data import load_ccf_template
        arr = load_ccf_template()

        self.assertEquals(arr.dtype, np.uint16)
        self.assertTupleEqual(arr.shape, (1320, 800, 1140))

    def test_load_allensdk_annotation(self):
        from neuralib.atlas.data import load_allensdk_annotation
        arr = load_allensdk_annotation()

        self.assertEquals(arr.dtype, np.uint32)
        self.assertTupleEqual(arr.shape, (1320, 800, 1140))

    def test_load_slice_view(self):
        from neuralib.atlas.view import load_slice_view
        arr = load_slice_view('ccf_annotation', 'sagittal').reference

        self.assertEquals(arr.dtype, np.uint16)
        self.assertTupleEqual(arr.shape, (1320, 800, 1140))


if __name__ == '__main__':
    unittest.main()
