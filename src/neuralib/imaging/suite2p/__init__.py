"""
Suite2p
===============================

.. code-block:: python

    from neuralib.imaging.suite2p import read_suite2p

    # Load Suite2p output directory (e.g., .../suite2p/plane0)
    result = read_suite2p("/path/to/suite2p/plane0")

    # Get shape of fluorescence data
    print(result.f_raw.shape)

    # Get number of neurons loaded based on threshold
    print(result.n_neurons)

    # Get GUI options
    print(result.ops)

    # Retrieve raw ROI coordinates
    rois_xy = result.get_rois_pixels()
    print("ROI coordinates (pixels):", rois_xy)


.. seealso::

    More methods and attrs available in: :class:`~neuralib.imaging.suite2p.core.Suite2PResult`



"""
from .core import *
from .plot import *
from .signals import *
