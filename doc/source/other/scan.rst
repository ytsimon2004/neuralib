Confocal Scan
======================

- This module provide the reading and parsing the scanning confocal image/metadata, currently support ``.lsm`` & ``.czi`` format.

- Use for batch processing/analysis, or other api integration.

- For general image visualization and processing, **Zen/Fiji/ImageJ** are easier and recommended


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




LSM Data
------------

- **Refer to API**: :doc:`../api/neuralib.scan.lsm`

.. code-block:: python

    from neuralib.scan.lsm import lsm_file

    filepath = ...
    with lsm_file(filepath) as lsm:
        print(lsm.metadata)  # get meta
        print(lsm.dimcode)  # get dim code
        print(lsm.n_scenes)  # get n scenes
        print(lsm.get_channel_names(scene=0))  # get channel names
        lsm.view(...)  # get the image array


.. warning::

    multi-scene and various dimensional lsm file are not supported and lack of testing yet.
    Contact the author if needed.


CZI Data
------------

- **Refer to API**: :doc:`../api/neuralib.scan.czi`

.. code-block:: python

    from neuralib.scan.lsm import czi_file

    filepath = ...
    with czi_file(filepath) as czi:
        print(czi.metadata)  # get meta
        print(czi.dimcode)  # get dim code
        print(czi.n_scenes)  # get n scenes
        print(czi.get_channel_names(scene=0))  # get channel names
        czi.view(...)  # get the image array

