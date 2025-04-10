"""
Suite2p Result
===============================

.. code-block:: python

    from neuralib.imaging.suite2p import read_suite2p

    # Load Suite2p output directory (e.g., .../suite2p/plane0)
    result = read_suite2p("/path/to/suite2p/plane0")

    # Get shape of fluorescence data
    print("Fluorescence shape:", result.f_raw.shape)

    # Get number of neurons loaded based on threshold
    print("Number of neurons:", result.n_neurons)

    # Check if dual-channel imaging was used
    if result.has_chan2:
        print("Red channel (chan2) available.")
        print("Number of red neurons:", result.n_red_neuron)

    # Retrieve raw ROI coordinates
    rois_xy = result.get_rois_pixels()
    print("ROI coordinates (pixels):", rois_xy)

    # Get neuron ID to raw index mapping
    mapping = result.get_neuron_id_mapping()
    print(mapping)

    # Get Suite2p coordinates scaled to mm
    from neuralib.imaging.suite2p import get_s2p_coords
    coords = get_s2p_coords(result, neuron_list=None, plane_index=0, factor=2.2)
    print(coords)


"""
from .core import *
from .plot import *
from .signals import *
