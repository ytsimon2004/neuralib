"""
Cellpose
=========

API Mode
---------


**Example of run an image segmentation in 2D mode and visualize using napari**

.. code-block:: console

    python -m neuralib.segmentation.cellpose.run_api -F <IMAGE_FILE> --napari

- It produces an output ``<IMAGE_FILE>_seg.npy`` in the same directory, If the output exists, the napari will directly use the output. If need run again and rewrite the output, use ``--force-eval`` option


**Example of run an image segmentation in 2D mode and visualize using Cellpose GUI**

.. code-block:: console

    python -m neuralib.segmentation.cellpose.run_api -F <IMAGE_FILE> --cpose



**Example of run batch image segmentation in 2D mode from a directory (.tif files)**

.. code-block:: console

    python -m neuralib.segmentation.cellpose.run_2d -D <DIRECTORY> --suffix .tif

- It produces multiple ``*_seg.npy`` in the same directory.


**See help using** ``-h`` **option**

.. code-block:: console

    python -m neuralib.segmentation.cellpose.run_api -h




Subprocess Mode
-----------------


**Example of run an image segmentation in subprocess call**

.. code-block:: console

    python -m neuralib.segmentation.cellpose.run_subproc -F <IMAGE_FILE>

- It produces an output ``<IMAGE_FILE>_seg.npy`` in the same directory, If the output exists, the napari will directly use the output. If need run again and rewrite the output, use ``--force-eval`` option



**Example of run batch image segmentation in subprocess call**

.. code-block:: console

    python -m neuralib.segmentation.cellpose.run_subproc -D <DIRECTORY>

- It produces multiple ``*.npy`` in the same directory.


**See help using** ``-h`` **option**

.. code-block:: console

    python -m neuralib.segmentation.cellpose.run_subproc -h


"""
