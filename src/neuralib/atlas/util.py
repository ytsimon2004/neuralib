from typing import Literal

import numpy as np

__all__ = ['ALLEN_CCF_10um_BREGMA',
           'ALLEN_SOURCE_TYPE',
           'PLANE_TYPE']

# allen CCF 10um volume coordinates, refer to allenCCF/Browsing Functions/allenCCFbregma.m
ALLEN_CCF_10um_BREGMA = np.array([540, 0, 570])  # AP, DV, LR
ALLEN_SOURCE_TYPE = Literal['annotation', 'nrrd', 'template']

PLANE_TYPE = Literal['coronal', 'sagittal', 'transverse']