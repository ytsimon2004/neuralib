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

.main([])

   pip install neura-library



Optional Modules
----------------

You can extend the functionality by installing optional modules:

Module-based
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Atlas Tools**:

.. code-block:: bash

   $ pip install "neura-library[atlas]"

**Scanner Support**:

.. code-block:: bash

   $ pip install "neura-library[scanner]"


**Imaging Utilities**:

.. code-block:: bash

   $ pip install "neura-library[imaging]"


**Image Processing Libraries**:

.. code-block:: bash

   $ pip install "neura-library[imglib]"


Functionality-based
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Rastermap: ``pip install "neura-library[rastermap]"``

- Cascade (TensorFlow): ``pip install "neura-library[cascade]"``

- StarDist: ``pip install "neura-library[stardist]"``


.. warning::
    For those heavy **Functionality-based** wrapper usage (i.e., Rastermap, Cascade, StarDist),
    Create a separate conda environment for the specific job is recommended (avoid dependencies conflict)

    For example you need to run **stardit**:

    .. code-block:: bash

        conda create -n neuralib_stardist python=3.10
        conda activate neuralib_stardist
        pip install neura-library[stardist]


All-In-One Installation
-----------------------

To install **everything**, including all optional features:

.. code-block:: bash

   $ pip install "neura-library[all]"

Development Installation
------------------------

For contributing and development:

.. code-block:: bash

   $ pip install "neura-library[dev]"

This installs all tools plus development utilities like `pre-commit` and `ruff`.

Documentation Build
-------------------

To build the documentation locally:

.. code-block:: bash

   $ pip install "neura-library[doc]"