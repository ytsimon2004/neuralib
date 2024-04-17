"""
ScanBox Data Parser
====================

:author:
    Yu-Ting Wei

This module simply provides

1. result parser for scanbox acquisition system

2. view the sequence output


See all the info
-------------------

.. code-block:: python

    from neuralib.calimg.scanbox import SBXInfo

    file = ...  # scanbox .mat output file
    sbx = SBXInfo.load(file)
    sbx.print_asdict()  # print the information as dictionary



Save as Json
-------------------

.. code-block:: python

   from neuralib.calimg.scanbox import sbx_to_json

   file = ...  # scanbox .mat output file
   output_file = ...  # *.json
   sbx_to_json(file, outputfile)




Screen Shot file to tiff
-------------------------

.. code-block:: python

    from neuralib.calimg.scanbox import screenshot_to_tiff

    file = ...  # scanbox .mat screenshot output file
    output = ... # *.tiff
    screenshot_to_tiff(file, output)



SBXViewer
==================

directly view the image sequence as mmap


Use CLI
------------

See help::

    python */scanbox/viewer.py -h


Example playing the 100-200 Frames::

    python */scanbox/viewer.py -D <DIR> -P <OPTIC_PLANE> -C <PMT_CHANNEL> -F 100,200


Example save 100-200 Frames as tiff::

    python */scanbox/viewer.py -D <DIR> -P <OPTIC_PLANE> -C <PMT_CHANNEL> -F 100,200 -O test.tiff



Use API call
--------------

.. code-block:: python

    from neuralib.calimg.scanbox.viewer import SBXViewer

    directory  = ...  # directory contain the .sbx and .mat output from scanbox
    sbx_viewer = SBXViewer(directory)

    # play
    sbx_viewer.play(slice(100,200), plane=0, channel=0)

    # save as tiff
    sbx_viewer.to_tiff(slice(100,200), plane=0, channel=0, output='test.tiff')


"""

from .core import *
from .viewer import *
