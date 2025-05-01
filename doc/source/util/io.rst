IO processing
==============
Module for handle different data input-output & project-specific cached files/examples


Helper Function
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - **Module**
     - **Description**
   * - :func:`~neuralib.io.csv_header.csv_header`
     - CSV context manager for writing with header control
   * - :class:`~neuralib.io.json.JsonEncodeHandler`
     - JSON reader/writer supporting extended data types
   * - :class:`~neuralib.io.pyh5.H5pyData`
     - HDF5 helper functions using `h5py`
   * - :func:`~neuralib.io.output.mkdir_version`
     - Save output files with version tracking


Example data
--------------
- **Refer to API**: :doc:`../api/neuralib.io.dataset`