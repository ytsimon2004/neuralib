"""
Tracking
============

Example usage of deeplabcut
---------------------------

TODO

Example usage of facemap
---------------------------

.. code-block:: python

    from neuralib.tracking import read_facemap

    # Load a facemap result directory
    result = read_facemap("/path/to/facemap/output")

    # Check if keypoint tracking data is available
    if result.with_keypoint:
        # list available keypoints
        print("Keypoints:", result.keypoints)

        # get data for a single keypoint
        df_eye = result.get("eye(back)").dataframe()
        print(df_eye)

        # get multiple keypoints and convert to z-scored coordinates
        df = result.get("eye(back)", "mouth").to_zscore()
        print(df)

    # Access pupil tracking data
    pupil_area = result.get_pupil_area()
    pupil_com = result.get_pupil_center_of_mass()

"""
from .deeplabcut import *
from .facemap import *
