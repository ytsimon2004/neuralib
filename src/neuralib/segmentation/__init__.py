"""
Segmentation
==============

Provide CLI for cellular segmentation and visualization, two packages are supported:

1. `Cellpose <https://github.com/MouseLand/cellpose>`_


2. `Stardist <https://github.com/stardist/stardist>`_

    **usage in 2D mode and visualize using napari**

    .. code-block:: console

        python -m neuralib.segmentation.stardist.run_2d -F <IMAGE_FILE> --napari

    it produces an json output ``<IMAGE_FILE>.json`` in the same directory.




"""
