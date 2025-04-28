Welcome to NeuraLib's documentation!
====================================


.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. contents:: Table of Contents
   :local:
   :depth: 2


API Reference
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Packages


   api/neuralib.rst


Installation
-----------------------

.. toctree::
   :maxdepth: 1
   :caption: Installation

   installation.rst



Release Notes
---------------

- Checkout `Release notes <https://github.com/ytsimon2004/neuralib/releases>`_


Modules
----------


:mod:`neuralib.atlas`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Module for slice view visualization, rois classification and whole brain reconstruction

- :mod:`neuralib.atlas.ccf`
    Customized dataframe for handling hierarchical classification of brain region/

- :mod:`neuralib.atlas.brainrender`
    CLI-based wrapper for `brainrender <https://github.com/brainglobe/brainrender>`_

- :mod:`neuralib.atlas.cellatlas`
    Volume and cell types counts for each brain region, refer to `Cell Atlas <https://portal.bluebrain.epfl.ch/resources/models/cell-atlas/>`_



:mod:`neuralib.imaging`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Module for 2-photon calcium imaging acquisition and result parsing

- :mod:`neuralib.imaging.scan_image`
    Data acquired from `ScanImage <https://www.mbfbioscience.com/products/scanimage/>`_ (under DEV)

- :mod:`neuralib.imaging.scanbox`
    Data acquired from `Scanbox <https://scanbox.org/tag/two-photon/>`_

- :mod:`neuralib.imaging.suite2p`
    Result parser for `suite2p <https://github.com/MouseLand/suite2p>`_

- :mod:`neuralib.imaging.spikes`
    dF/F to spike activity (OASIS/Cascade)

- :mod:`neuralib.imaging.widefield`
    Wide-field image sequence processing


:mod:`neuralib.segmentation`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Module for neuronal segmentation

- :mod:`neuralib.segmentation.cellpose`
    Result parser and batch running for `cellpose <https://github.com/MouseLand/cellpose>`_

- :mod:`neuralib.segmentation.stardist`
    Result parser and batch running for `stardist <https://github.com/stardist/stardist>`_


:mod:`neuralib.locomotion`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Module for handle animal's locomotion

- :mod:`neuralib.locomotion.epoch`
    Selection of specific epoch (i.e., running, stationary)

- :mod:`neuralib.locomotion.position`
    Position in environment, current only support 1D circular (i.e., linear treadmill)


:mod:`neuralib.model`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :mod:`neuralib.model.bayes_decoding`
    Position decoding using population neuronal activity

- :mod:`neuralib.model.rastermap`
    Run and result parser for `rastermap <https://github.com/MouseLand/rastermap>`_



:mod:`neuralib.morpho`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Morphological reconstruction data presentation



:mod:`neuralib.tracking`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :mod:`neuralib.tracking.deeplabcut`
    Result parser for `DeepLabCut <https://github.com/DeepLabCut/DeepLabCut>`_

- :mod:`neuralib.tracking.facemap`
    Result parser for `facemap <https://github.com/MouseLand/facemap>`_


Utilities Modules
-------------------


:mod:`neuralib.argp`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Use argparse as dataclass field


:mod:`neuralib.persistence`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Caching the analyzed results (i.e., concatenation for statistic purpose)


:mod:`neuralib.sqlp`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Python functions to build a SQL (sqlite3) statement.


:mod:`neuralib.dashboard`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Interactive dashboard visualization


:mod:`neuralib.plot`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Module for general plotting purpose


:mod:`neuralib.imglib`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Image processing library


:mod:`neuralib.io`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- File IO and example dataset


:mod:`neuralib.tools.gspread`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Google spreadsheet API wrapper for read/write


:mod:`neuralib.tools.slack_bot`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Real-time Slack notification bot for analysis pipeline


:mod:`neuralib.util`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Utility module


CLI project.scripts
---------------------------

neuralib_brainrender
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- See examples in `api <https://neuralib.readthedocs.io/en/latest/api/neuralib.atlas.brainrender.html>`_

.. prompt:: bash $

    neuralib_brainrender -h


neuralib_widefield
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- See example in  `api <https://neuralib.readthedocs.io/en/latest/api/neuralib.imaging.widefield.html>`_

.. prompt:: bash $

    neuralib_widefield -h



Doc for Array type
-------------------

- ``Array[DType, [*Shape]]``
- ``DType`` = array datatype. ``Shape`` = array shape. ``|`` = Union

**Example**

- ``int`` or ``bool`` with (N,3) array -> ``Array[int|bool, [N, 3]]``
- ``float`` array with union shape (N,2) or (N,T,2) -> ``Array[float, [N, 2]|[N, T, 2]]``


Notebook Example
-------------------

.. toctree::
    :maxdepth: 1
    :caption: Notebook

    ../notebooks/example_imaging_2p

    ../notebooks/example_rastermap

    ../notebooks/example_segmentation
    ../notebooks/example_slice_view
    ../notebooks/example_neuralib_plot


.. toctree::
    :maxdepth: 2
    :caption: Roadmap

    roadmap.rst




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`