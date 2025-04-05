"""
ROI DataFrame Classes
=====================

This module provides tools to structure, normalize, and analyze region-of-interest (ROI) data in brain atlases.
It supports classification based on Allen Brain hierarchy, hemisphere computation, normalization (volume, cell count, or channel fraction), and subregion breakdown.

RoiClassifierDataFrame
----------------------

A dataframe where each row is an ROI annotated by anatomical and experimental metadata.

**Required columns**

- ``acronym``: Brain area acronym (e.g. 'VISp', 'RSPd')
- ``AP_location``: Anterior-posterior coordinate in mm
- ``DV_location``: Dorsal-ventral coordinate in mm
- ``ML_location``: Medial-lateral coordinate in mm
- ``channel``: Fluorescent channel (e.g. 'gfp', 'rfp', 'overlap')
- ``source``: Experimental source or injection site

**Example**

.. code-block:: python

    import polars as pl
    from neuralib.atlas.ccf.classifier import RoiClassifierDataFrame

    df = pl.DataFrame({
        "acronym": ["RSPd", "RSPd", "VISp", "VISp"],
        "AP_location": [1.2, 1.3, -2.4, -2.6],
        "DV_location": [1.0, 1.1, 2.0, 2.1],
        "ML_location": [0.4, -0.3, 0.2, -0.2],
        "channel": ["gfp", "gfp", "rfp", "rfp"],
        "source": ["aRSC", "aRSC", "pRSC", "pRSC"]
    })

    roi = RoiClassifierDataFrame(df)
    print(roi)

.. code-block:: text

    ┌────────┬────────────┬────────────┬────────────┬────────┬────────┐
    │acronym ┆AP_location ┆DV_location ┆ML_location ┆channel ┆source  │
    │ ---    ┆ ---        ┆ ---        ┆ ---        ┆ ---    ┆ ---    │
    │ str    ┆ f64        ┆ f64        ┆ f64        ┆ str    ┆ str    │
    ╞════════╪════════════╪════════════╪════════════╪════════╪════════╡
    │ RSPd   ┆ 1.2        ┆ 1.0        ┆ 0.4        ┆ gfp    ┆ aRSC   │
    │ RSPd   ┆ 1.3        ┆ 1.1        ┆ -0.3       ┆ gfp    ┆ aRSC   │
    │ VISp   ┆ -2.4       ┆ 2.0        ┆ 0.2        ┆ rfp    ┆ pRSC   │
    │ VISp   ┆ -2.6       ┆ 2.1        ┆ -0.2       ┆ rfp    ┆ pRSC   │
    └────────┴────────────┴────────────┴────────────┴────────┴────────┘


RoiNormalizedDataFrame
----------------------

Returned by ``.to_normalized()``. Represents ROIs aggregated and normalized by region and source.

**Columns depend on normalization mode** (volume, cell, or channel):

- ``source``: Injection source (e.g. 'aRSC')
- ``tree_2``: Brain region (based on hierarchy level)
- ``counts``: Raw counts
- ``fraction``: % of ROIs for each source
- ``hemisphere``: Hemisphere label
- ``normalized``: Normalized value

**Example**

.. code-block:: python

    norm = roi.to_normalized(
        norm="volume",
        level=2,
        hemisphere="both"
    )
    print(norm)

.. code-block:: text

    ┌─────────┬────────┬────────┬───────────┬────────────┬────────────────┬────────────┐
    │ source  ┆ tree_2 ┆ counts ┆ fraction  ┆ hemisphere ┆ Volumes [mm^3] ┆ normalized │
    │ ---     ┆ ---    ┆ ---    ┆ ---       ┆ ---        ┆ ---            ┆ ---        │
    │ str     ┆ str    ┆ u32    ┆ f64       ┆ str        ┆ f64            ┆ f64        │
    ╞═════════╪════════╪════════╪═══════════╪════════════╪════════════════╪════════════╡
    │ overlap ┆ ACA    ┆ 1208   ┆ 29.997517 ┆ both       ┆ 5.222484       ┆ 231.307537 │
    │ pRSC    ┆ ACA    ┆ 3296   ┆ 22.822324 ┆ both       ┆ 5.222484       ┆ 631.117254 │
    │ …       ┆ …      ┆ …      ┆ …         ┆ …          ┆ …              ┆ …          │
    │ pRSC    ┆ VIS    ┆ 4035   ┆ 27.939344 ┆ both       ┆ 12.957203      ┆ 311.409797 │
    │ overlap ┆ VIS    ┆ 628    ┆ 15.594736 ┆ both       ┆ 12.957203      ┆ 48.46725   │
    │ aRSC    ┆ VIS    ┆ 3865   ┆ 12.627005 ┆ both       ┆ 12.957203      ┆ 298.289682 │
    └─────────┴────────┴────────┴───────────┴────────────┴────────────────┴────────────┘


RoiSubregionDataFrame
---------------------

Returned by ``.to_subregion(region)``. Breaks down a major region (e.g. 'VIS') into subregion contributions per source.

**Rows**: sources

**Columns**: subregion acronyms (e.g. VISam, VISp, VISal, ...)

**Example**

.. code-block:: python

    sub = roi.to_subregion("VIS")
    print(sub)

.. code-block:: text

    ┌─────────┬───────────┬───────────┬───────────┬───┬──────────┬──────────┬──────────┬──────────┐
    │ source  ┆ VISam     ┆ VISp      ┆ VISpm     ┆ … ┆ VISal    ┆ VISpor   ┆ VISli    ┆ VISpl    │
    │ ---     ┆ ---       ┆ ---       ┆ ---       ┆   ┆ ---      ┆ ---      ┆ ---      ┆ ---      │
    │ str     ┆ f64       ┆ f64       ┆ f64       ┆   ┆ f64      ┆ f64      ┆ f64      ┆ f64      │
    ╞═════════╪═══════════╪═══════════╪═══════════╪═══╪══════════╪══════════╪══════════╪══════════╡
    │ overlap ┆ 39.649682 ┆ 15.127389 ┆ 28.025478 ┆ … ┆ 3.025478 ┆ 2.707006 ┆ 1.592357 ┆ 0.159236 │
    │ aRSC    ┆ 32.160414 ┆ 28.952135 ┆ 23.05304  ┆ … ┆ 6.080207 ┆ 1.293661 ┆ 2.069858 ┆ 0.07762  │
    │ pRSC    ┆ 25.947955 ┆ 27.95539  ┆ 27.459727 ┆ … ┆ 3.122677 ┆ 2.973978 ┆ 1.982652 ┆ 1.016109 │
    └─────────┴───────────┴───────────┴───────────┴───┴──────────┴──────────┴──────────┴──────────┘

**Profile Table**

Also accessible via ``sub.profile``:

.. code-block:: text

    ┌─────────┬────────┬───────┬────────────────┐
    │ source  ┆ counts ┆ total ┆ total_fraction │
    ╞═════════╪════════╪═══════╪════════════════╡
    │ aRSC    ┆ 3865   ┆ 30609 ┆ 0.12627        │
    │ pRSC    ┆ 4035   ┆ 14442 ┆ 0.279393       │
    │ overlap ┆ 628    ┆ 4027  ┆ 0.155947       │
    └─────────┴────────┴───────┴────────────────┘

"""

from .dataframe import *
from .matrix import *
