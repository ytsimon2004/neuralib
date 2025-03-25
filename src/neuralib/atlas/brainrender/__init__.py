"""
BrainRender Wrapper
===================

:author:
    Yu-Ting Wei

This module provide a CLI-based `brainrender <https://brainglobe.info/documentation/brainrender/index.html.>`_ wrapper
Can be run as command line once the package installed

See the available options use ``-h`` option ::

    neuralib_brainrender <area | roi | probe> -h


Region reconstruction
---------------------------------------

Example of reconstruct the Visual Cortex ::

    neuralib_brainrender area -R VISal,VISam,VISl,VISli,VISp,VISpl,VISpm,VISpor --camera top


|brender area|



ROI reconstruction
---------------------------------------

By default use CCF coordinates space, specify use [--coord-space] option {ccf,brainrender}


numpy file
^^^^^^^^^^^^^

Example of numpy file (``Array[float, [N, 3]]`` with AP, DV, ML coordinates) ::

    [[-3.03  4.34 -4.5 ]
     [-3.03  4.42 -4.37]
     [-3.03  4.55 -4.37]
     ...
     [-2.91  4.31  4.75]
     [-2.91  4.36  4.77]
     [-2.91  4.12  4.85]]


.. code-block:: console

    neuralib_brainrender roi --file <NUMPY_FILE>

csv file
^^^^^^^^^^^^^

Example of csv file (with ``AP_location``, ``DV_location``, ``ML_location`` headers) ::

    ┌─────────────┬─────────────┬─────────────┐
    │ AP_location ┆ DV_location ┆ ML_location │
    │ ---         ┆ ---         ┆ ---         │
    │ f64         ┆ f64         ┆ f64         │
    ╞═════════════╪═════════════╪═════════════╡
    │ -3.03       ┆ 4.34        ┆ -4.5        │
    │ -3.03       ┆ 4.92        ┆ -4.31       │
    │ …           ┆ …           ┆ …           │
    │ -2.91       ┆ 4.06        ┆ 4.71        │
    │ -2.91       ┆ 4.12        ┆ 4.85        │
    └─────────────┴─────────────┴─────────────┘


.. code-block:: console

    neuralib_brainrender roi --file <CSV_FILE>




processed csv file (flexible reconstruction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO

Example of using parsed allenccf csv output ::

    ┌───────────────────────────────────┬─────────┬─────────────┬─────────────┬─────────────┬─────────┬─────────┬────────┬────────────┐
    │ name                              ┆ acronym ┆ AP_location ┆ DV_location ┆ ML_location ┆ avIndex ┆ channel ┆ source ┆  ...       │
    │ ---                               ┆ ---     ┆ ---         ┆ ---         ┆ ---         ┆ ---     ┆ ---     ┆ ---    ┆  ---       │
    │ str                               ┆ str     ┆ f64         ┆ f64         ┆ f64         ┆ i64     ┆ str     ┆ str    ┆  ...       │
    ╞═══════════════════════════════════╪═════════╪═════════════╪═════════════╪═════════════╪═════════╪═════════╪════════╪════════════╡
    │ Ectorhinal area/Layer 5           ┆ ECT5    ┆ -3.03       ┆ 4.34        ┆ -4.5        ┆ 377     ┆ gfp     ┆ aRSC   ┆  ...       │
    │ Perirhinal area layer 6a          ┆ PERI6a  ┆ -3.03       ┆ 4.42        ┆ -4.37       ┆ 372     ┆ gfp     ┆ aRSC   ┆  ...       │
    │ …                                 ┆ …       ┆ …           ┆ …           ┆ …           ┆ …       ┆ …       ┆ …      ┆  …         │
    │ Ventral auditory area layer 6a    ┆ AUDv6a  ┆ -2.91       ┆ 3.52        ┆ 4.46        ┆ 156     ┆ rfp     ┆ pRSC   ┆  ...       │
    │ Ectorhinal area/Layer 6a          ┆ ECT6a   ┆ -2.91       ┆ 4.14        ┆ 4.47        ┆ 378     ┆ rfp     ┆ pRSC   ┆  ...       │
    │ Temporal association areas layer… ┆ TEa5    ┆ -2.91       ┆ 4.02        ┆ 4.55        ┆ 365     ┆ rfp     ┆ pRSC   ┆  ...       │
    └───────────────────────────────────┴─────────┴─────────────┴─────────────┴─────────────┴─────────┴─────────┴────────┴────────────┘


.. code-block:: console

    neuralib_brainrender roi --classifier-file <CSV_FILE>


|brender roi|








Probe reconstruction (probe mode)
---------------------------------------

Reconstruct the probe (or shanks) in accordance with trajectory labeling (e.g., DiI, DiO or lesion track...)

Prepare CSV file from ccf pipeline::

    ┌───────────────────────────────────┬─────────┬─────────────┬─────────────┬─────────────┬─────────┐
    │ name                              ┆ acronym ┆ AP_location ┆ DV_location ┆ ML_location ┆ avIndex │
    │ ---                               ┆ ---     ┆ ---         ┆ ---         ┆ ---         ┆ ---     │
    │ str                               ┆ str     ┆ f64         ┆ f64         ┆ f64         ┆ i64     │
    ╞═══════════════════════════════════╪═════════╪═════════════╪═════════════╪═════════════╪═════════╡
    │ Primary visual area layer 6a      ┆ VISp6a  ┆ -3.81       ┆ 1.92        ┆ -3.12       ┆ 191     │
    │ optic radiation                   ┆ or      ┆ -4.08       ┆ 2.33        ┆ -3.12       ┆ 1217    │
    │ Posterolateral visual area layer… ┆ VISpl6a ┆ -4.28       ┆ 2.29        ┆ -3.12       ┆ 198     │
    │ Posterolateral visual area layer… ┆ VISpl5  ┆ -4.52       ┆ 2.17        ┆ -3.12       ┆ 197     │
    │ Subiculum                         ┆ SUB     ┆ -3.93       ┆ 4.36        ┆ -3.3        ┆ 536     │
    │ Entorhinal area medial part dors… ┆ ENTm5   ┆ -4.19       ┆ 4.39        ┆ -3.3        ┆ 515     │
    │ Entorhinal area medial part dors… ┆ ENTm2   ┆ -4.44       ┆ 4.39        ┆ -3.3        ┆ 510     │
    │ Entorhinal area medial part dors… ┆ ENTm1   ┆ -4.66       ┆ 4.29        ┆ -3.3        ┆ 509     │
    └───────────────────────────────────┴─────────┴─────────────┴─────────────┴─────────────┴─────────┘

- Row number equal to shank numbers * 2 (in the example, use the 4 shanks NeuroPixel probe), 2 points indicate the most dorsal & ventral detected signals on the serial brain slices

- Use ``-P`` to specify the slice cutting orientation {coronal,sagittal,transverse}. If multiple shanks were inserted along the AP axis, assume do the sagittal plane, if inserted along the ML axis, then assume the coronal plane

- Assume shanks are not bended to perform interpolation based on the insert depth ``-D``



Example of Above csv file for targeting the left Entorhinal cortex (ENT) using 4 shanks NeuroPixel probe::

    neuralib_brainrender probe -F <CSV_FILE> --depth 3000 -P sagittal -R ENT -H left




|brender probe|




See the available options use ``-h`` option ::

    neuralib_brainrender probe -h




.. |brender area| image:: ../_static/brender_area.png
.. |brender roi| image:: ../_static/brender_roi.png
.. |brender probe| image:: ../_static/brender_probe.png


"""
