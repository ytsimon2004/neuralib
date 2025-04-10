Installation
============

.. toctree::
    :maxdepth: 1

The **neura-library** provides a flexible and modular installation pipeline for a wide range of tools used in systems neuroscience research.

Requirements
------------

- Python >= 3.10
- Supported OS: OS Independent
- Recommended: virtual environment (e.g. `uv`, `conda`)

Basic Installation
------------------

To install the core functionality of the library:

.. prompt:: bash $

   pip install neura-library



Optional Modules
----------------

You can extend the functionality by installing optional modules:

Module-based
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Atlas Tools**:

.. prompt:: bash $

   pip install "neura-library[atlas]"

**Scanner Support**:

.. prompt:: bash $

   pip install "neura-library[scanner]"


**Imaging Utilities**:

.. prompt:: bash $

   pip install "neura-library[imaging]"


**Image Processing Libraries**:

.. prompt:: bash $

   pip install "neura-library[imagelib]"


Functionality-based
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Rastermap: ``pip install "neura-library[rastermap]"``

- Cascade (TensorFlow): ``pip install "neura-library[cascade]"``

- Cellpose: ``pip install "neura-library[cellpose]"``

- StarDist: ``pip install "neura-library[stardist]"``

- Slack integration: ``pip install "neura-library[slack]"``

- Google Sheets API: ``pip install "neura-library[gspread]"``


.. warning::
    For those heavy **Functionality-based** wrapper usage (i.e., Rastermap, Cascade, Cellpose, StarDist),
    Create a separate conda environment for the specific job is recommended (avoid dependencies conflict)

    For example you need to run **cellpose**:

    .. prompt:: bash $

        conda create -n neuralib_seg python=3.10
        conda activate neuralib_seg
        pip install neura-library[cellpose]


All-In-One Installation
-----------------------

To install **everything**, including all optional features:

.. prompt:: bash $

   pip install "neura-library[all]"

Development Installation
------------------------

For contributing and development:

.. prompt:: bash $

   pip install "neura-library[dev]"

This installs all tools plus development utilities like `pre-commit` and `ruff`.

Documentation Build
-------------------

To build the documentation locally:

.. prompt:: bash $

   pip install "neura-library[doc]"