"""
Facemap Wrapper
===========================

This module provide the result parser facemap
Including keypoints and pupil diameter data


Example of load the pupil area
--------------------------------

.. code-block:: python

    from neuralib.wrapper.facemap import *

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
TODO



"""
from .core import *
