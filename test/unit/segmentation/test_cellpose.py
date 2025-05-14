import subprocess
import sys

import pytest

from neuralib.io import NEUROLIB_DATASET_DIRECTORY
from neuralib.io.dataset import load_example_rois_image
from neuralib.segmentation.cellpose.core import cellpose_point_roi_helper

try:
    import cellpose as cp
except ImportError:
    cp = None


@pytest.mark.skipif(cp is None, reason="cellpose is not installed")
def test_seg_to_roi():
    load_example_rois_image(cached=True, rename_file='rois.png')
    file = NEUROLIB_DATASET_DIRECTORY / 'rois.png'

    cmds = [sys.executable, '-m', 'cellpose']
    cmds.extend(['--image_path', str(file)])
    cmds.extend(['--use_gpu'])

    result = subprocess.run(cmds, capture_output=True, text=True)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

    cellpose_point_roi_helper(file.with_stem(file.stem + '_seg').with_suffix('.npy'), file.with_suffix('.roi'))
