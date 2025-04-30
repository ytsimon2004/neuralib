Model
========
Module for algorithm computing

Bayesian Position decoding
----------------------------
Position decoding using population neuronal activity

- **Refer to API**: :mod:`neuralib.model.bayes_decoding`


RasterMap
---------------------

RasterMap is an unsupervised discovery algorithm for neural data, developed by
Carsen Stringer and Marius Pachitariu. It is particularly useful for identifying
spatial or temporal patterns in large-scale neural recordings.

- **Refer to API**: :mod:`neuralib.model.rastermap`

This module provides the following functions:

- ``run``: Perform RasterMap embedding on neural or imaging data.
- ``plot``: Visualize RasterMap outputs, optionally alongside behavioral measurements.
- ``read``: Load previously saved RasterMap results.



Supported Input Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Cellular datasets** (e.g., electrophysiology spikes or calcium imaging):

  ``Array[float, [N, T]]``

  - ``N``: Number of neurons
  - ``T``: Number of timepoints or samples

- **Widefield imaging datasets**:

  ``Array[Any, [T, H, W]]``

  - ``T``: Number of timepoints
  - ``H``: Image height (pixels)
  - ``W``: Image width (pixels)


References
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `GitHub repository <https://github.com/MouseLand/rastermap>`_
- `Example notebook (cellular data) <https://colab.research.google.com/github/MouseLand/rastermap/blob/main/notebooks/rastermap_largescale.ipynb>`_
- `Example notebook (widefield data) <https://colab.research.google.com/github/MouseLand/rastermap/blob/main/notebooks/rastermap_widefield.ipynb>`_
