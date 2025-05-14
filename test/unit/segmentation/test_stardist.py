import unittest
from pathlib import Path

import numpy as np

from argclz.core import parse_args
from neuralib.io import NEUROLIB_DATASET_DIRECTORY
from neuralib.io.dataset import load_example_rois_image, load_example_rois_dir
from neuralib.segmentation.stardist.run_2d import StarDist2DOptions


class TestStarDist(unittest.TestCase):
    arr: np.ndarray
    dirpath: Path
    filepath: Path

    @classmethod
    def setUpClass(cls):
        load_example_rois_image(cached=True, rename_file='rois.png')
        cls.filepath = NEUROLIB_DATASET_DIRECTORY / 'rois.png'
        cls.dirpath = load_example_rois_dir(cached=True, rename_folder='rois')

    def test_empty_option(self):
        opt = parse_args(StarDist2DOptions(), [])
        with self.assertRaises(RuntimeError):
            opt.run()

    def test_file_run(self, test_napari: bool = False):
        args = ['--file', str(self.filepath)]
        if test_napari:
            args.append('--napari')

        opt = parse_args(StarDist2DOptions(), args)
        self.assertTrue(np.issubdtype(opt.process_image().dtype, np.floating))
        self.assertTrue(opt.file_mode)
        opt.run()

    def test_dir_mode(self):
        opt = parse_args(StarDist2DOptions(), ['--dir', str(self.dirpath), '--invalid', '--save_roi'])
        self.assertTrue(opt.batch_mode)
        opt.run()

    # @classmethod
    # def tearDownClass(cls):
    #     clean_all_cache_dataset()
