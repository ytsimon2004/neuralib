"""
BrainRender Wrapper
===================

:author:
    Yu-Ting Wei

This module provide a CLI-based brainrender wrapper
See detail in the https://brainglobe.info/documentation/brainrender/index.html.
The wrapper provide three main usage cases, and can be run as command line once the package installed

Region reconstruction (area mode)
---------------------------------------

Plot brain regions

Example of reconstruct the Visual Cortex ::

    brender area -R VISal,VISam,VISl,VISli,VISp,VISpl,VISpm,VISpor --camera top


|brender area|


See the available options use `-h` option ::

    brender area -h



ROI reconstruction (roi mode)
---------------------------------------

Plot brain regions with ROIs label

Example of reconstruct ROIs in the Somatosensory Cortex for ipsilateral hemisphere(assume right hemisphere)::

    brender roi -F <CSV_FILE>


|brender roi|



CSV FILE example (auto transformed coordinates space from allen to brainrender)::

    ┌───────────────────────────────────┬─────────┬─────────────┬─────────────┬─────────────┬─────────┬─────────┬────────┬───────────────────────────┬──────────────┬────────┬────────────┬────────────┬────────────┬────────────┬────────────┬───────────┐
    │ name                              ┆ acronym ┆ AP_location ┆ DV_location ┆ ML_location ┆ avIndex ┆ channel ┆ source ┆ abbr                      ┆ acronym_abbr ┆ hemi.  ┆ merge_ac_0 ┆ merge_ac_1 ┆ merge_ac_2 ┆ merge_ac_3 ┆ merge_ac_4 ┆ family    │
    │ ---                               ┆ ---     ┆ ---         ┆ ---         ┆ ---         ┆ ---     ┆ ---     ┆ ---    ┆ ---                       ┆ ---          ┆ ---    ┆ ---        ┆ ---        ┆ ---        ┆ ---        ┆ ---        ┆ ---       │
    │ str                               ┆ str     ┆ f64         ┆ f64         ┆ f64         ┆ i64     ┆ str     ┆ str    ┆ str                       ┆ str          ┆ str    ┆ str        ┆ str        ┆ str        ┆ str        ┆ str        ┆ str       │
    ╞═══════════════════════════════════╪═════════╪═════════════╪═════════════╪═════════════╪═════════╪═════════╪════════╪═══════════════════════════╪══════════════╪════════╪════════════╪════════════╪════════════╪════════════╪════════════╪═══════════╡
    │ Ectorhinal area/Layer 5           ┆ ECT5    ┆ -3.03       ┆ 4.34        ┆ -4.5        ┆ 377     ┆ gfp     ┆ aRSC   ┆ Ectorhinal area           ┆ ECT          ┆ contra ┆ ECT        ┆ ECT        ┆ ECT        ┆ ECT        ┆ ECT        ┆ ISOCORTEX │
    │ Perirhinal area layer 6a          ┆ PERI6a  ┆ -3.03       ┆ 4.42        ┆ -4.37       ┆ 372     ┆ gfp     ┆ aRSC   ┆ Perirhinal area           ┆ PERI         ┆ contra ┆ PERI       ┆ PERI       ┆ PERI       ┆ PERI       ┆ PERI       ┆ ISOCORTEX │
    │ …                                 ┆ …       ┆ …           ┆ …           ┆ …           ┆ …       ┆ …       ┆ …      ┆ …                         ┆ …            ┆ …      ┆ …          ┆ …          ┆ …          ┆ …          ┆ …          ┆ …         │
    │ Ventral auditory area layer 6a    ┆ AUDv6a  ┆ -2.91       ┆ 3.52        ┆ 4.46        ┆ 156     ┆ rfp     ┆ pRSC   ┆ Ventral auditory area     ┆ AUDv         ┆ ipsi   ┆ AUD        ┆ AUD        ┆ AUD        ┆ AUD        ┆ AUDv       ┆ ISOCORTEX │
    │ Ectorhinal area/Layer 6a          ┆ ECT6a   ┆ -2.91       ┆ 4.14        ┆ 4.47        ┆ 378     ┆ rfp     ┆ pRSC   ┆ Ectorhinal area           ┆ ECT          ┆ ipsi   ┆ ECT        ┆ ECT        ┆ ECT        ┆ ECT        ┆ ECT        ┆ ISOCORTEX │
    │ Temporal association areas layer… ┆ TEa5    ┆ -2.91       ┆ 4.02        ┆ 4.55        ┆ 365     ┆ rfp     ┆ pRSC   ┆ Temporal association area ┆ TEa          ┆ ipsi   ┆ TEa        ┆ TEa        ┆ TEa        ┆ TEa        ┆ TEa        ┆ ISOCORTEX │
    └───────────────────────────────────┴─────────┴─────────────┴─────────────┴─────────────┴─────────┴─────────┴────────┴───────────────────────────┴──────────────┴────────┴────────────┴────────────┴────────────┴────────────┴────────────┴───────────┘

See how to create the csv after ccf pipeline

.. code-block:: python

    from neuralib.atlas.ccf.classifier import RoiClassifier
    from neuralib.atlas.ccf.core import AbstractCCFDir
    root = ...
    ccf_dir = AbstractCCFDir(root, with_overlap_sources=False)
    classifier = RoiClassifier(ccf_dir, plane='coronal')
    df = classifier.parsed_df

Example ccf data folder structure in (:class:`~neuralib.atlas.ccf.core.AbstractCCFDir()`)



See the available options use ``-h`` option ::

    brender roi -h



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

    brender probe -F <CSV_FILE> -D 3000 -P sagittal -R ENT -H left




|brender probe|




See the available options use ``-h`` option ::

    brender probe -h




.. |brender area| image:: ../_static/brender_area.png
.. |brender roi| image:: ../_static/brender_roi.png
.. |brender probe| image:: ../_static/brender_probe.png


"""

from .core import *
from .probe_rst import *
from .roi_rst import *
