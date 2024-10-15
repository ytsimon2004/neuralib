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

   api/neuralib.rst


Release Notes
---------------

- Checkout `Release notes <https://github.com/ytsimon2004/neuralib/releases>`_


Installation
-----------------------

Install common dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- First run for the common dependencies (Python>=3.9, but >=3.11 not yet tested)

.. code-block:: console

    pip install neura-library


Install all dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    pip install neura-library[all]

Install the minimal required dependencies according to usage purpose
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Choices in ``[]``: ``atlas``, ``scanner``, ``calimg``, ``segmentation``, ``model``, ``track``, ``gpu``, ``profile``, ``imagelib``, ``tools``, ``full``

- Example of using ``atlas`` module:

.. code-block:: console

    pip install neura-library[atlas]

- Example of using ``segmentation`` module:

.. code-block:: console

    pip install neura-library[segmentation]


Install pre-commit and linter check (For developer only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    pip install neura-library[dev]

-  Setup pre-commit by:

.. code-block:: console

    pre-commit install

-  Do dry run ``ruff`` lint check by:

.. code-block:: console

    ruff check .


CLI project.scripts
---------------------------

.. code-block:: console

    brender -h


- See examples in `api <https://neuralib.readthedocs.io/en/latest/api/neuralib.atlas.brainrender.html>`_



Notebook Demo
------------------------------

.. toctree::
    :maxdepth: 1

    ../notebooks/example_calimg

    ../notebooks/example_facemap
    ../notebooks/example_rastermap_2p
    ../notebooks/example_rastermap_wfield

    ../notebooks/example_segmentation
    ../notebooks/example_slice_view
    ../notebooks/example_ibl_plot
    ../notebooks/example_neuralib_plot


Modules
----------


:mod:`neuralib.atlas`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Module for whole brain, slice view visualization and rois classification

    - :mod:`neuralib.atlas.ccf`: Customized hierarchical classification for the mouse brain atlas

    - :mod:`neuralib.atlas.brainrender`: CLI-based wrapper for `brainrender <https://github.com/brainglobe/brainrender>`_

    - :mod:`neuralib.atlas.cellatlas`: Volume and cell types counts for each brain region, refer to `Cell Atlas <https://portal.bluebrain.epfl.ch/resources/models/cell-atlas/>`_

    - :mod:`neuralib.atlas.ibl`: Slice view plotting wrapper for `ibllib <https://github.com/int-brain-lab/ibllib?tab=readme-ov-file>`_ and `iblatlas <https://int-brain-lab.github.io/iblenv/_autosummary/ibllib.atlas.html>`_


:mod:`neuralib.calimg`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Module for 2-photon calcium imaging acquisition and result parsing

    - :mod:`neuralib.calimg.scan_image`: Data acquired from `ScanImage <https://www.mbfbioscience.com/products/scanimage/>`_ (under DEV)

    - :mod:`neuralib.calimg.scanbox`: Data acquired from `Scanbox <https://scanbox.org/tag/two-photon/>`_

    - :mod:`neuralib.calimg.suite2p`: Result parser for `suite2p <https://github.com/MouseLand/suite2p>`_

    - :mod:`neuralib.calimg.spikes`: dF/F to spike activity (OASIS/Cascade)


:mod:`neuralib.segmentation`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Module for neuronal segmentation

    - :mod:`neuralib.segmentation.cellpose`: Result parser and batch running for `cellpose <https://github.com/MouseLand/cellpose>`_

    - :mod:`neuralib.segmentation.stardist`: Result parser and batch running for `stardist <https://github.com/stardist/stardist>`_


:mod:`neuralib.locomotion`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Module for handle animal's locomotion

    - :mod:`neuralib.locomotion.epoch`: Selection of specific epoch (i.e., running, stationary)

    - :mod:`neuralib.locomotion.position`: Position in environment, current only support 1D circular (i.e., linear treadmill)



:mod:`neuralib.model`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    - :mod:`neuralib.model.bayes_decoding`: Position decoding using population neuronal activity

    - :mod:`neuralib.model.rastermap`: Run and result parser for `rastermap <https://github.com/MouseLand/rastermap>`_



:mod:`neuralib.tracking`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    - :mod:`neuralib.tracking.deeplabcut`: Result parser for `DeepLabCut <https://github.com/DeepLabCut/DeepLabCut>`_

    - :mod:`neuralib.tracking.facemap`: Result parser for `facemap <https://github.com/MouseLand/facemap>`_


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


:mod:`neuralib.util.cli_args`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    - Run script as subprocess


:mod:`neuralib.util.color_logging`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    - Logging with color format


:mod:`neuralib.util.gpu`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    - OS-dependent GPU info


:mod:`neuralib.util.profile`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    - Simple benchmark profile testing and debugging


:mod:`neuralib.util.table`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    - Rich table visualization


:mod:`neuralib.util.segments`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    - Epoch or value segments



Doc for Array type
-------------------

- ``Array[DType, [*Shape]]``
- ``DType`` = array datatype. ``Shape`` = array shape. ``|`` = Union

**Example**

- ``int`` or ``bool`` with (N,3) array -> ``Array[int|bool, [N, 3]]``
- ``float`` array with union shape (N,2) or (N,T,2) -> ``Array[float, [N, 2]|[N, T, 2]]``



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`