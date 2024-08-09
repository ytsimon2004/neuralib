"""
StarDist
=========


**Example of run an image segmentation in 2D mode and visualize using napari**

.. code-block:: console

    python -m neuralib.segmentation.stardist.run_2d -F <IMAGE_FILE> --napari

- It produces an output ``<IMAGE_FILE>.npz`` in the same directory, If the output exists, the napari will directly use the output. If need run again and rewrite the output, use ``--force-eval`` option



**Example of run batch image segmentation in 2D mode from a directory (.tif files)**

.. code-block:: console

    python -m neuralib.segmentation.stardist.run_2d -D <DIRECTORY> --suffix .tif

- It produces multiple ``*.npz`` in the same directory.


**See help using** ``-h`` **option**

.. code-block:: console

    python -m neuralib.segmentation.stardist.run_2d -h

"""
