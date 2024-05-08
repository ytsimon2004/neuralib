"""
CellAtlas Dataframe
====================

cell atlas about cell types and volume information for each brain area

.. seealso:: `<https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010739#sec047>`_

Example of loading the dataframe
----------------------------------

.. code-block:: python

    from neuralib.atlas.cellatlas import CellAtlas
    catlas = CellAtlas.load_from_csv()
    print(catlas.dataframe)

output::

    ┌────────────────────────────────┬────────────────┬───────────┐
    │ Brain region                   ┆ Volumes [mm^3] ┆ n_neurons │
    │ ---                            ┆ ---            ┆ ---       │
    │ str                            ┆ f64            ┆ i64       │
    ╞════════════════════════════════╪════════════════╪═══════════╡
    │ Abducens nucleus               ┆ 0.015281       ┆ 1324      │
    │ Accessory facial motor nucleus ┆ 0.013453       ┆ 497       │
    │ Accessory olfactory bulb       ┆ 0.6880625      ┆ 189608    │
    │ …                              ┆ …              ┆ …         │
    │ Zona incerta                   ┆ 2.157641       ┆ 136765    │
    │ posteromedial visual area      ┆ 1.2225625      ┆ 197643    │
    └────────────────────────────────┴────────────────┴───────────┘


Example of loading the sync dataframe between cellatlas/allen brain acronym
------------------------------------------------------------------------------

.. code-block:: python

    from neuralib.atlas.cellatlas import CellAtlas
    print(catlas.load_sync_allen_structure_tree())


output::

    ┌────────────────────────────────┬────────────────┬───────────┬─────────┐
    │ name                           ┆ Volumes [mm^3] ┆ n_neurons ┆ acronym │
    │ ---                            ┆ ---            ┆ ---       ┆ ---     │
    │ str                            ┆ f64            ┆ i64       ┆ str     │
    ╞════════════════════════════════╪════════════════╪═══════════╪═════════╡
    │ Abducens nucleus               ┆ 0.015281       ┆ 1324      ┆ VI      │
    │ Agranular insular area         ┆ 4.901734       ┆ 242362    ┆ AI      │
    │ …                              ┆ …              ┆ …         ┆ …       │
    │ Visual areas                   ┆ 12.957203      ┆ 1297194   ┆ VIS     │
    │ Zona incerta                   ┆ 2.157641       ┆ 136765    ┆ ZI      │
    │ posteromedial visual area      ┆ 1.2225625      ┆ 197643    ┆ VISpm   │
    └────────────────────────────────┴────────────────┴───────────┴─────────┘

"""
from .core import *
