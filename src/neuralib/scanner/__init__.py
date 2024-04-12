"""
Confocal Scanner
======================

- This module provide the reading and parsing the Zeiss confocal dataset,
currently support .lsm & .czi format.

- Use for batch processing/analysis, or other api integration.
(i.e., batch cell segmentation, manually stitching...)

- For general image visualization, **Zen/Fiji/ImageJ** are recommended




Image Data Dim
---------------

`Dimension parameters (DimCode)`:

    V - view

    H - phase

    I - illumination

    S - scene

    R - rotation

    T - time

    C - channel

    Z - z plane (height)

    M - mosaic tile, mosaic images only

    Y - image height

    X - image width

    A - samples, BGR/RGB images only




LSM format
------------

.. code-block:: python

    from neuralib.scanner import LSMConfocalScanner

    filepath = ...
    lsm = LSMConfocalScanner.load(filepath)

    # get the meta
    print(lsm.meta)

    # get the images array
    print(lsm.lsmfile)

    # zproj imshow
    lsm.imshow(channel=2, zproj_type='max')



CZI format
------------

.. code-block:: python

    from neuralib.scanner import LSMConfocalScanner

    filepath = ...
    czi = CziConfocalScanner.load(filepath)

    # get meta
    print(czi.meta)

    # get dim code
    print(czi.get_dim_code())

    # get the image array
    print(czi.get_image(channel=0, scene=1))

    # zproj imshow
    lsm.imshow()

"""
from .core import *
from .czi import *
from .lsm import *
