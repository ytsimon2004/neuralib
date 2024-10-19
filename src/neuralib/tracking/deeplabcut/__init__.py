"""
Deeplabcut
=============

This module provide ``Deeplabcut`` result parsing
Including model meta, and output .h5 or .csv


Example of load the results
-----------------------------------

.. code-block:: python

    from neuralib.tracking.deeplabcut import *

    file = ...  # .h5 or .csv file path
    meta = ... # .pickle meta file path
    dlc = load_dlc_result(file, meta).with_global_lh_filter(lh=0.95)  # with likelihood filter >= 0.95

    # Get polars dataframe
    print(dlc.dat)

    # See all the joints
    print(dlc.joints)

    # Get a xy numpy array from specific joint. i.e., labeled Nose
    print(dlc['Nose'].xy)



Example of load the meta typeddict
-----------------------------------

.. code-block:: python

    from neuralib.tracking.deeplabcut import *

    file = ...  # .h5 or .csv file path
    meta = ... # .pickle meta file path
    dlc = load_dlc_result(file, meta).with_global_lh_filter(lh=0.95)  # with likelihood filter >= 0.95

    # See meta DeepLabCutMeta typeddict
    print(dlc.meta)

    # See model_config typeddict
    print(dlc.meta['model_config'])



"""
from .core import *
