"""
ScanBox Data Parser
====================

:author:
    Yu-Ting Wei

This module simply provide the
1. result parser for scanbox acquisition system
2. view the sequence output (under DEV)


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
------------------
directly view the image sequence as mmap
TODO


"""

from .core import *
from .viewer import *
