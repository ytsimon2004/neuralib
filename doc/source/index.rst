.. NeuraLib documentation master file, created by
   sphinx-quickstart on Fri Mar 29 16:52:36 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NeuraLib's documentation!
====================================

Utility tools for system neuroscience research, including Open Source Wrapper or Parser

Modules including:

- ``argp``: Use argparse as dataclass field
- ``atlas``: Whole brain, slice view visualization and rois classification
- ``bokeh_model``: Interactive dashboard visualization
- ``calimg``: Module for 2photon calcium imaging acquisition and result parsing
- ``persistence``: Module for caching the analyzed results (i.e., concatenation for statistic purpose)
- ``plot``: Module for general plotting purpose
- ``scanner``: Module for parsing Zeiss confocal scanning data
- ``segmentation``: Module for cellular segmentation
- ``stimpy``: Tools and Result parser for visual-guided behavior dataset (lab internal use)
- ``tools``: other small tools...
- ``wrapper``: Module for other open-source tools wrapper, including deeplabcut, facemap, rastermap
- ``util``: ...


.. toctree::
   :maxdepth: 2
   :caption: Contents:

API Reference
-----------------

.. toctree::
   :maxdepth: 1

   api/neuralib.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
