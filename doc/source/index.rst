.. NeuraLib documentation master file, created by
   sphinx-quickstart on Fri Mar 29 16:52:36 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NeuraLib's documentation!
====================================



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

Installation
-----------------------

.. code-block:: console

   pip install neura-library

- Checkout `Release notes <https://github.com/ytsimon2004/neuralib/releases>`_


Open-Source tools API call / data parsing
------------------------------------------

atlas
^^^^^

- Module for whole brain, slice view visualization and rois classification

  - ``neuralib.atlas.ccf``: Customized hierarchical classification for the mouse brain atlas

  - ``neuralib.atlas.brainrender``: CLI-based wrapper for `brainrender <https://github.com/brainglobe/brainrender>`_

  - ``neuralib.atlas.cellatlas``: Volume and cell types counts for each brain region, refer to `Cell Atlas <https://portal.bluebrain.epfl.ch/resources/models/cell-atlas/>`_

  - ``neuralib.atlas.ibl``: Slice view plotting wrapper for `ibllib <https://github.com/int-brain-lab/ibllib?tab=readme-ov-file>`_ and `iblatlas <https://int-brain-lab.github.io/iblenv/_autosummary/ibllib.atlas.html>`_

calimg
^^^^^^

- Module for 2-photon calcium imaging acquisition and result parsing

  - ``neuralib.calimg.scan_image``: Data acquired from `ScanImage <https://www.mbfbioscience.com/products/scanimage/>`_ (under DEV)

  - ``neuralib.calimg.scanbox``: Data acquired from `Scanbox <https://scanbox.org/tag/two-photon/>`_

  - ``neuralib.calimg.suite2p``: Result parser for `suite2p <https://github.com/MouseLand/suite2p>`_

segmentation
^^^^^^^^^^^^

- Module for neuronal segmentation

  - ``neuralib.segmentation.cellpose``: Result parser and batch running for `cellpose <https://github.com/MouseLand/cellpose>`_

  - ``neuralib.segmentation.stardist``: Result parser and batch running for `stardist <https://github.com/stardist/stardist>`_

wrapper
^^^^^^^

- Module for other open-source tools wrapper

  - ``neuralib.wrapper.deeplabcut``: Result parser for `DeepLabCut <https://github.com/DeepLabCut/DeepLabCut>`_

  - ``neuralib.wrapper.facemap``: Result parser for `facemap <https://github.com/MouseLand/facemap>`_

  - ``neuralib.wrapper.rastermap``: Run and result parser for `rastermap <https://github.com/MouseLand/rastermap>`_


Utilities Modules
------------------

argp
^^^^

- ``neuralib.argp``: Use argparse as dataclass field

persistence
^^^^^^^^^^^

- ``neuralib.persistence``: Caching the analyzed results (i.e., concatenation for statistic purpose)

bokeh_model
^^^^^^^^^^^

- ``neuralib.bokeh_model``: Interactive dashboard visualization

sqlp
^^^^

- ``neuralib.sqlp``: Python functions to build a SQL (sqlite3) statement.

Others
-------

- ``neuralib.plot``: Module for general plotting purpose
- ``neuralib.model.bayes_decoding``: Position decoding using population neuronal activity
- ``neuralib.tools.imglib``: Image processing library (under DEV)
- ``neuralib.tools.slack_bot``: Real-time Slack notification bot for analysis pipeline
- ``neuralib.util.cli_args``: Run script as subprocess
- ``neuralib.util.color_logging``: Logging with color format
- ``neuralib.util.csv``: CSV context manager
- ``neuralib.util.gpu``: OS-dependent GPU info
- ``neuralib.util.profile_test``: Simple benchmark profile testing
- ``neuralib.util.table``: Rich table visualization


project.scripts using CLI
---------------------------

- ``brender``: See examples in `api <https://neuralib.readthedocs.io/en/latest/api/neuralib.atlas.brainrender.html>`_




.. toctree::
    :maxdepth: 1
    :caption: Notebook Demo

    ../notebooks/example_calimg

    ../notebooks/example_facemap
    ../notebooks/example_rastermap_2p
    ../notebooks/example_rastermap_wfield

    ../notebooks/example_slice_view
    ../notebooks/example_ibl_plot
    ../notebooks/example_neuralib_plot


