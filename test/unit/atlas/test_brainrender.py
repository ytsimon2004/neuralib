from pathlib import Path

import pytest

from neuralib.atlas.brainrender.probe import ProbeRenderCLI
from neuralib.atlas.brainrender.roi import RoiRenderCLI
from neuralib.io.dataset import google_drive_file

DATA_EXISTS = (Path().home() / '.brainglobe' / 'allen_mouse_10um_v1.2').exists()


@pytest.mark.skipif(not DATA_EXISTS, reason="source data need to be downloaded")
def test_brainrender_rois():
    with google_drive_file('1cf2r3kcqjENBQMe8YzBQvol8tZgscN4J', rename_file='classifier.csv') as f:
        class Test(RoiRenderCLI):
            classifier_file = f

        Test().main()


@pytest.mark.skipif(not DATA_EXISTS, reason="source data need to be downloaded")
def test_brainrender_npx2():
    with google_drive_file('1fRvMNHhGgh5KP3CgGm6CMFth1qIAmwfh', rename_file='probe.csv') as f:
        class Test(ProbeRenderCLI):
            file = f
            implant_depth = 3000
            plane_type = 'sagittal'
            regions = ('ENT',)
            hemisphere = 'left'

        Test().main()
