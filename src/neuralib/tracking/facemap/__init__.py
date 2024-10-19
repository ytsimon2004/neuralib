"""
Facemap
==========

This module provide ``Facemap`` result parsing
Including keypoints and pupil diameter data


Example of load the pupil area
--------------------------------

.. code-block:: python

    from neuralib.tracking.facemap import *

    directory = ...  # directory with output *proc.npy
    fmap = FaceMapResult.load(directory, track_type='pupil')
    area = fmap.get_pupil_area()



Example of launch the GUI with the existed results
--------------------------------------------------------

.. code-block:: python

    from neuralib.wrapper.facemap import *

    directory = ...  # directory with output *proc.npy (pathlike)
    with_keypoints = False  # if have keypoint tracking
    env_name = ...  # conda env name with facemap package installed (str)
    FaceMapResult.launch_facemap_gui(directory, with_keypoints=with_keypoints, env_name=env_name)



Example of load the keypoint result
------------------------------------

.. code-block:: python

    directory = ...
    fmap = FaceMapResult.load(directory, track_type='keypoints')

    plot_facemap_keypoints(fmap, frame_interval=(0, 100), keypoints=['eye(back)', 'eye(bottom)', 'eye(front)', 'eye(top)'])



.. toctree::
    :maxdepth: 1
    :caption: Notebook Demo

    ../notebooks/example_facemap

"""
from .core import *
