"""
Atlas CCF
===============

:author:
    Yu-Ting Wei


This module provide analysis pipeline for the data acquired from allenccf mouse brain registration tool

specifically for the ROIs distribution across the whole brain

.. seealso:: `<https://github.com/cortex-lab/allenCCF>`_


Data Folder structure should follow::

    ANIMAL_001/ (root)
        ├── raw/ (optional)
        ├── zproj/
        │    └── ANIMAL_001_g*_s*_{channel}.tif
        ├── roi/
        │    └── ANIMAL_001_g*_s*_{channel}.roi
        ├── roi_cpose/
        │    └── ANIMAL_001_g*_s*_{channel}.roi
        ├── resize/ (src for the allenccf)
        │    ├── ANIMAL_001_g*_s*_resize.tif
        │    └── processed/
        │           ├── ANIMAL_001_g*_s*_resize_processed.tif
        │           └── transformations/
        │                 ├── ANIMAL_001_g*_s*_resize_processed_transformed.tif
        │                 ├── ANIMAL_001_g*_s*_resize_processed_transform_data.mat
        │                 └── labelled_regions/
        │                       ├── {*channel}_roitable.csv
        │                       └── parsed_data /
        │                             └── parsed_csv_merge.csv
        │
        └── output_files/ (for generate output fig)



Example Raw data csv (``{*channel}_roitable.csv`` in the folder structure)::

    ┌───────────────────────────────────┬─────────┬─────────────┬─────────────┬─────────────┬─────────┐
    │ name                              ┆ acronym ┆ AP_location ┆ DV_location ┆ ML_location ┆ avIndex │
    │ ---                               ┆ ---     ┆ ---         ┆ ---         ┆ ---         ┆ ---     │
    │ str                               ┆ str     ┆ f64         ┆ f64         ┆ f64         ┆ i64     │
    ╞═══════════════════════════════════╪═════════╪═════════════╪═════════════╪═════════════╪═════════╡
    │ Entorhinal area lateral part lay… ┆ ENTl5   ┆ -4.35       ┆ 3.98        ┆ 4.23        ┆ 504     │
    │ Entorhinal area lateral part lay… ┆ ENTl3   ┆ -4.35       ┆ 4.09        ┆ 4.25        ┆ 501     │
    │ Entorhinal area lateral part lay… ┆ ENTl2   ┆ -4.35       ┆ 3.42        ┆ 4.44        ┆ 497     │
    │ Entorhinal area lateral part lay… ┆ ENTl2   ┆ -4.35       ┆ 3.56        ┆ 4.46        ┆ 497     │
    │ Primary visual area layer 6a      ┆ VISp6a  ┆ -4.23       ┆ 1.65        ┆ -2.28       ┆ 191     │
    │ …                                 ┆ …       ┆ …           ┆ …           ┆ …           ┆ …       │
    │ Laterointermediate area layer 4   ┆ VISli4  ┆ -3.54       ┆ 2.04        ┆ 4.01        ┆ 210     │
    │ Temporal association areas layer… ┆ TEa6b   ┆ -3.54       ┆ 2.89        ┆ 4.02        ┆ 367     │
    │ alveus                            ┆ alv     ┆ -3.54       ┆ 3.52        ┆ 4.14        ┆ 1246    │
    │ Field CA1                         ┆ CA1     ┆ -3.54       ┆ 3.61        ┆ 4.15        ┆ 458     │
    │ optic radiation                   ┆ or      ┆ -3.54       ┆ 3.77        ┆ 4.25        ┆ 1217    │
    └───────────────────────────────────┴─────────┴─────────────┴─────────────┴─────────────┴─────────┘


Parsing Pipeline
-------------------

Do the following procedure

1. concat roi from different channels
2. add column(fields) for user-specific manner. e.g., 'channel', 'source', ...
3. converge to hierarchical level / family based on allen brain region tree (Wang et al 2020).
   see :meth:`~neuralib.atlas.map.merge_until_level()` in `neuralib.atlas.map.py <https://github.com/ytsimon2004/neuralib/blob/main/src/neuralib/atlas/map.py>`_


.. code-block:: python

    from neuralib.atlas.ccf.classifier import UserInjectionConfig, RoiClassifier
    from neuralib.atlas.ccf.core import AbstractCCFDir

    # prepare ccf folder structure
    root = ...
    ccf_dir = AbstractCCFDir(root, with_overlap_sources=True)  # assume count overlap channel rois separately

    # example of injection configuration
    USER_CONFIG = UserInjectionConfig(
        area='RSP',
        hemisphere='ipsi',
        ignore=True,
        fluor_repr=dict(
            rfp='pRSC',
            gfp='aRSC',
            overlap='overlap'
        )
    )

    classifier = RoiClassifier(ccf_dir, merge_level=2, plane='coronal', config=USER_CONFIG)
    print(classifier.parsed_df)


output::

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


Do the ROIs normalization
---------------------

three normalization provided (:class:`~neuralib.atlas.ccf.norm.MouseBrainRoiNormHandler`)

1. ``channel``: normalize to fraction of rois for a specific color fluorescence channel

2. ``volume``: normalize to the volume size per region (cellatlas-based)

3. ``cell``: normalize to the total cell counts per region (cellatlas-based)


.. code-block:: python

    from neuralib.atlas.ccf.classifier import RoiClassifiedNormTable
    from neuralib.atlas.ccf.norm import MouseBrainRoiNormHandler

    norm = MouseBrainRoiNormHandler(norm_type='volume')

    # use the classifier constructed above
    norm_data: RoiClassifiedNormTable = classifier.get_classified_data(norm)
    print(norm_data.data)


output::

    ┌─────────┬────────────┬────────┬───────────┬───┬───────────┬───────────┬─────────────────┬────────┐
    │ channel ┆ merge_ac_2 ┆ n_rois ┆ percent   ┆ … ┆ Volumes   ┆ n_neurons ┆ *volume_norm_n_r┆ animal │
    │ ---     ┆ ---        ┆ ---    ┆ ---       ┆   ┆ [mm^3]    ┆ ---       ┆ ois             ┆ ---    │
    │ str     ┆ str        ┆ i64    ┆ f64       ┆   ┆ ---       ┆ i64       ┆ ---             ┆ str    │
    │         ┆            ┆        ┆           ┆   ┆ f64       ┆           ┆ f64             ┆        │
    ╞═════════╪════════════╪════════╪═══════════╪═══╪═══════════╪═══════════╪═════════════════╪════════╡
    │ overlap ┆ ACA        ┆ 423    ┆ 30.344333 ┆ … ┆ 5.222484  ┆ 337372    ┆ 80.995934       ┆ YW051  │
    │ gfp     ┆ MO         ┆ 3352   ┆ 24.545987 ┆ … ┆ 22.248234 ┆ 985411    ┆ 150.663641      ┆ YW051  │
    │ rfp     ┆ ACA        ┆ 1383   ┆ 23.791502 ┆ … ┆ 5.222484  ┆ 337372    ┆ 264.816494      ┆ YW051  │
    │ gfp     ┆ ACA        ┆ 3130   ┆ 22.920328 ┆ … ┆ 5.222484  ┆ 337372    ┆ 599.331616      ┆ YW051  │
    │ …       ┆ …          ┆ …      ┆ …         ┆ … ┆ …         ┆ …         ┆ …               ┆ …      │
    │ overlap ┆ SS         ┆ 1      ┆ 0.071736  ┆ … ┆ 37.177937 ┆ 2384622   ┆ 0.026898        ┆ YW051  │
    │ overlap ┆ ECT        ┆ 1      ┆ 0.071736  ┆ … ┆ 3.457703  ┆ 387378    ┆ 0.289209        ┆ YW051  │
    │ overlap ┆ TEa        ┆ 1      ┆ 0.071736  ┆ … ┆ 3.860953  ┆ 386396    ┆ 0.259003        ┆ YW051  │
    │ rfp     ┆ TT         ┆ 1      ┆ 0.017203  ┆ … ┆ 1.734078  ┆ 124596    ┆ 0.576675        ┆ YW051  │
    └─────────┴────────────┴────────┴───────────┴───┴───────────┴───────────┴─────────────────┴────────┘



Do the ROI subregions query
-----------------

Example for query the ROIs in visual cortex

.. code-block:: python

    from neuralib.atlas.ccf.query import RoiAreaQuery

    # use the classifier constructed above
    df = classifier.parsed_df
    result = RoiAreaQuery.by(df, 'VIS).get_subregion_result
    print(result.data)


output::

    ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬────────┐
    │ source  ┆ VISam   ┆ VISp    ┆ VISpm   ┆ VISl    ┆ VISal   ┆ VISpor  ┆ VISli   ┆ VISpl   ┆ VISC   │
    │ ---     ┆ ---     ┆ ---     ┆ ---     ┆ ---     ┆ ---     ┆ ---     ┆ ---     ┆ ---     ┆ ---    │
    │ str     ┆ f64     ┆ f64     ┆ f64     ┆ f64     ┆ f64     ┆ f64     ┆ f64     ┆ f64     ┆ f64    │
    ╞═════════╪═════════╪═════════╪═════════╪═════════╪═════════╪═════════╪═════════╪═════════╪════════╡
    │ aRSC    ┆ 25.4669 ┆ 34.0318 ┆ 23.0979 ┆ 6.42369 ┆ 6.37813 ┆ 1.77676 ┆ 2.27790 ┆ 0.0     ┆ 0.5466 │
    │         ┆ 7       ┆ 91      ┆ 5       ┆         ┆ 2       ┆ 5       ┆ 4       ┆         ┆ 97     │
    │ pRSC    ┆ 15.1445 ┆ 36.1079 ┆ 29.5568 ┆ 10.7129 ┆ 3.04431 ┆ 2.38921 ┆ 1.92678 ┆ 1.07899 ┆ 0.0385 │
    │         ┆ 09      ┆         ┆ 4       ┆ 09      ┆ 6       ┆         ┆ 2       ┆ 8       ┆ 36     │
    │ overlap ┆ 48.5294 ┆ 8.82352 ┆ 31.25   ┆ 4.04411 ┆ 3.30882 ┆ 2.57352 ┆ 1.10294 ┆ 0.0     ┆ 0.3676 │
    │         ┆ 12      ┆ 9       ┆         ┆ 8       ┆ 4       ┆ 9       ┆ 1       ┆         ┆ 47     │
    └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴────────┘



"""
