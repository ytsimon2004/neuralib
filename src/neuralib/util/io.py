"""
Default Path for Cache
====================

.. code-block:: python

    CACHE_DIRECTORY = Path.home() / '.cache'
    NEUROLIB_CACHE_DIRECTORY = CACHE_DIRECTORY / 'neuralib'

    # ATLAS
    ATLAS_CACHE_DIRECTORY = NEUROLIB_CACHE_DIRECTORY / 'atlas'
    CCF_CACHE_DIRECTORY = ATLAS_CACHE_DIRECTORY / 'ccf_2017'
    ALLEN_SDK_DIRECTORY = ATLAS_CACHE_DIRECTORY / 'allensdk'
    IBL_CACHE_DIRECTORY = ATLAS_CACHE_DIRECTORY / 'ibl'


Example of default cache under the ~/.cache/neuralib
-----------------------------------------------------
::

    neuralib
        └── atlas
            ├── allensdk
            │         ├── annotation_10.nrrd
            │         ├── manifest.json
            │         └── structures.json
            ├── ccf_2017
            │         ├── annotation_volume_10um_by_index.npy
            │         ├── structure_tree_safe_2017.csv
            │         └── template_volume_10um.npy
            ├── cellatlas.csv
            ├── cellatlas_allen_sync.csv
            └── ibl
                ├── annotation_10.nrrd
                ├── annotation_10_lut_v01.npz
                └── average_template_10.nrrd





"""
from pathlib import Path

__all__ = [
    'CACHE_DIRECTORY',
    'NEUROLIB_CACHE_DIRECTORY',
    #
    'ATLAS_CACHE_DIRECTORY',
    'CCF_CACHE_DIRECTORY',
    'ALLEN_SDK_DIRECTORY',
    'IBL_CACHE_DIRECTORY'
]

CACHE_DIRECTORY = Path.home() / '.cache'
NEUROLIB_CACHE_DIRECTORY = CACHE_DIRECTORY / 'neuralib'

# ATLAS
ATLAS_CACHE_DIRECTORY = NEUROLIB_CACHE_DIRECTORY / 'atlas'
CCF_CACHE_DIRECTORY = ATLAS_CACHE_DIRECTORY / 'ccf_2017'
ALLEN_SDK_DIRECTORY = ATLAS_CACHE_DIRECTORY / 'allensdk'
IBL_CACHE_DIRECTORY = ATLAS_CACHE_DIRECTORY / 'ibl'
