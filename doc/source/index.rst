Welcome to NeuraLib's documentation!
====================================
**NeuraLib** is a utility toolkit designed for rodent systems neuroscience research.
It provides wrappers, parsers, and tools for efficient data handling, analysis, and reproducibility within open-source neuroscience workflows.

Key Features
------------

- **Utility tools for rodent neuroscience experiments**
- **Open-source parsers and wrappers**
- **Lightweight, modular design for easy integration**
- **Clean documentation and comprehensive API reference**

Resources
---------

- `GitHub Repository <https://github.com/ytsimon2004/neuralib>`_
- `Release Notes <https://github.com/ytsimon2004/neuralib/releases>`_

Installation
------------

To install NeuraLib via pip:

.. code-block:: bash

    $ pip install neura-library

For more detailed instructions, see :doc:`installation`.

Getting Started
---------------

.. toctree::
   :maxdepth: 3
   :caption: Atlas

   atlas/index

.. toctree::
   :maxdepth: 3
   :caption: Imaging

   imaging/index

.. toctree::
   :maxdepth: 3
   :caption: Other

   other/index

.. toctree::
   :maxdepth: 3
   :caption: Utility

   util/index

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/neuralib.rst

Command-Line Tools
------------------

neuralib_brainrender
^^^^^^^^^^^^^^^^^^^^

- Visualize brain region data with built-in rendering support
- See examples in the :doc:`atlas/brainrender`

.. code-block:: bash

    $ neuralib_brainrender -h

neuralib_widefield
^^^^^^^^^^^^^^^^^^

- Widefield imaging CLI analysis
- See examples in the :doc:`imaging/widefield`

.. code-block:: bash

    $ neuralib_widefield -h

Array Annotation Syntax
-----------------------

Used in documentation to describe array-shaped data structures:

- ``Array[DType, [*Shape]]`` where:
  - ``DType`` = data type (e.g., `int`, `float`, `bool`)
  - ``Shape`` = array shape (e.g., `[N, T]`)
  - ``|`` = denotes a union of shapes or types

**Examples:**

- Boolean or integer array with shape `(N, 3)`:

  ``Array[int|bool, [N, 3]]``

- Float array with shape `(N, 2)` or `(N, T, 2)`:

  ``Array[float, [N, 2] | [N, T, 2]]``

.. toctree::
   :maxdepth: 2
   :caption: Roadmap

   roadmap.rst

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
