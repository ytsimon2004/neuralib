"""
CellAtlas Dataframe
====================

Cell types and Volume information for each brain area

.. seealso::

    - `Blue Brain Cell Atlas <https://bbp.epfl.ch/nexus/cell-atlas/doc/about.html>`_

    - `Rodarie D et al., (2022) <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010739#sec047>`_

Example of loading the dataframe
----------------------------------

.. code-block:: python

    from neuralib.atlas.cellatlas import load_cellatlas

    df = load_cellatlas()
    print(df)


**output** ::

    ┌────────────────────────────────┬────────────────┬───────────┬─────────┐
    │ name                           ┆ Volumes [mm^3] ┆ n_neurons ┆ acronym │
    │ ---                            ┆ ---            ┆ ---       ┆ ---     │
    │ str                            ┆ f64            ┆ i64       ┆ str     │
    ╞════════════════════════════════╪════════════════╪═══════════╪═════════╡
    │ Abducens nucleus               ┆ 0.015281       ┆ 1324      ┆ VI      │
    │ Accessory facial motor nucleus ┆ 0.013453       ┆ 497       ┆ ACVII   │
    │ Accessory olfactory bulb       ┆ 0.6880625      ┆ 189608    ┆ AOB     │
    │ Accessory supraoptic group     ┆ 0.0013125      ┆ 148       ┆ ASO     │
    │ Agranular insular area         ┆ 4.901734       ┆ 242362    ┆ AI      │
    │ …                              ┆ …              ┆ …         ┆ …       │
    │ Vestibular nuclei              ┆ 2.5563125      ┆ 87832     ┆ VNC     │
    │ Visceral area                  ┆ 1.764797       ┆ 108294    ┆ VISC    │
    │ Visual areas                   ┆ 12.957203      ┆ 1297194   ┆ VIS     │
    │ Zona incerta                   ┆ 2.157641       ┆ 136765    ┆ ZI      │
    │ posteromedial visual area      ┆ 1.2225625      ┆ 197643    ┆ VISpm   │
    └────────────────────────────────┴────────────────┴───────────┴─────────┘




"""
from .core import *
