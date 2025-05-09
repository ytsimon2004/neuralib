Cellular Segmentation
=======================


Provide CLI for cellular segmentation and visualization

Design mainly for quick visualization of results and batch processing (i.e., whole-brain image), and save as imageJ/Fiji ``.roi``.


**Supported packages**

    1. `Cellpose <https://github.com/MouseLand/cellpose>`_

    - See usage :mod:`neuralib.segmentation.cellpose`

    2. `Stardist <https://github.com/stardist/stardist>`_

    - See usage in :mod:`neuralib.segmentation.stardist`

    If only small amount of images, **the native GUI are recommended**.


StarDist
-------------


**Example of run an image segmentation in 2D mode and visualize using napari**

.. prompt:: bash $

    python -m neuralib.segmentation.stardist.run_2d -F <IMAGE_FILE> --napari

- It produces an output ``<IMAGE_FILE>.npz`` in the same directory, If the output exists, the napari will directly use the output. If need run again and rewrite the output, use ``--force-eval`` option



**Example of run batch image segmentation in 2D mode from a directory (.tif files)**

.. prompt:: bash $

    python -m neuralib.segmentation.stardist.run_2d -D <DIRECTORY> --suffix .tif

- It produces multiple ``*.npz`` in the same directory.


**See help using** ``-h`` **option**

.. prompt:: bash $

    python -m neuralib.segmentation.stardist.run_2d -h



Cellpose
------------------

API Mode
^^^^^^^^^^^^^^^^^^^^^^^^


**Example of run an image segmentation in 2D mode and visualize using napari**

.. prompt:: bash $

    python -m neuralib.segmentation.cellpose.run_api -F <IMAGE_FILE> --napari

- It produces an output ``<IMAGE_FILE>_seg.npy`` in the same directory, If the output exists, the napari will directly use the output. If need run again and rewrite the output, use ``--force-eval`` option


**Example of run an image segmentation in 2D mode and visualize using Cellpose GUI**

.. prompt:: bash $

    python -m neuralib.segmentation.cellpose.run_api -F <IMAGE_FILE> --cpose



**Example of run batch image segmentation in 2D mode from a directory (.tif files)**

.. prompt:: bash $

    python -m neuralib.segmentation.cellpose.run_2d -D <DIRECTORY> --suffix .tif

- It produces multiple ``*_seg.npy`` in the same directory.


**See help using** ``-h`` **option**

.. prompt:: bash $

    python -m neuralib.segmentation.cellpose.run_api -h




Subprocess Mode
^^^^^^^^^^^^^^^^^^^^^^^^


**Example of run an image segmentation in subprocess call**

.. prompt:: bash $

    python -m neuralib.segmentation.cellpose.run_subproc -F <IMAGE_FILE>

- It produces an output ``<IMAGE_FILE>_seg.npy`` in the same directory, If the output exists, the napari will directly use the output. If need run again and rewrite the output, use ``--force-eval`` option



**Example of run batch image segmentation in subprocess call**

.. prompt:: bash $

    python -m neuralib.segmentation.cellpose.run_subproc -D <DIRECTORY>

- It produces multiple ``*.npy`` in the same directory.


**See help using** ``-h`` **option**

.. prompt:: bash $

    python -m neuralib.segmentation.cellpose.run_subproc -h