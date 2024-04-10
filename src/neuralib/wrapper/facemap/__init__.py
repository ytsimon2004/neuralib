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
    frame_time = ...  # video time
    fmap = FaceMapResult.load(directory, track_type='pupil', frame_time=frame_time)
    area = fmap.get_pupil_tracking()['area']



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
from .util import *
