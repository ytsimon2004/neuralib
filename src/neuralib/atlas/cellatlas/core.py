from __future__ import annotations

import dataclasses
import io
from pathlib import Path

import pandas as pd
import polars as pl

from neuralib.atlas.data import load_structure_tree
from neuralib.util.io import ATLAS_CACHE_DIRECTORY
from neuralib.util.util_verbose import fprint

__all__ = ['CellAtlas']


@dataclasses.dataclass
class CellAtlas:
    dataframe: pl.DataFrame
    """
    Example::
    
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
    """

    @classmethod
    def load_from_csv(cls,
                      file: Path | None = None,
                      ignore_cell_types_info: bool = True,
                      ignore_detail_info: bool = True) -> CellAtlas:
        """
        Load/Download the csv file

        :param file: filepath. If None, download from source paper
        :param ignore_cell_types_info: ignore cell types information, only select neuron and volume foreach areas
        :param ignore_detail_info: ignore information in brain subregion
        :return:
        """
        if file is None:
            d = ATLAS_CACHE_DIRECTORY

            if not d.exists():
                d.mkdir(exist_ok=True, parents=True)

            file = d / 'cellatlas.csv'
            if not file.exists():
                cls._request(file)

        df = pl.read_csv(file)

        if ignore_cell_types_info:
            df = df.select('Brain region', 'Neuron [mm^-3]', 'Volumes [mm^3]')

        # total neurons
        df = (
            df.with_columns((pl.col('Neuron [mm^-3]') * pl.col('Volumes [mm^3]')).alias('n_neurons').cast(pl.Int64))
            .drop('Neuron [mm^-3]')
        )

        if ignore_detail_info:
            patterns = (',', '/', r'\(')
            for pt in patterns:
                df = df.filter(~(pl.col('Brain region').str.contains(pt)))

        return CellAtlas(df.sort('Brain region'))

    @classmethod
    def _request(cls, output: Path) -> pl.DataFrame:
        """download from paper source"""
        import requests

        url = 'https://journals.plos.org/ploscompbiol/article/file?type=supplementary&id=10.1371/' \
              'journal.pcbi.1010739.s011'
        resp = requests.get(url)

        if resp.status_code == 200:
            df = pd.read_excel(io.BytesIO(resp.content), sheet_name='Densities BBCAv1')
            pl.from_pandas(df).write_csv(output)
            fprint(f'Download successfully cellatlas csv and save in {output}!', vtype='io')
        else:
            raise RuntimeError('download cellatlas FAIL')

        return df

    @property
    def brain_regions(self) -> list[str]:
        """list of brain regions"""
        return self.dataframe['Brain region'].unique().to_list()

    @classmethod
    def load_sync_allen_structure_tree(cls, force_save: bool = True) -> pl.DataFrame:
        """
        TODO `ProS` not found
        Based on cellatlas dataframe, create a sync used `acronym` header in allen struct_tree (sorted by name)

        **fields**

        ``name``: `Brain region` found in cellatlas

        ``n_neurons``: `n_neurons` from cellatlas dataframe

        ``acronym``: `acronym` found in the structure_tree csv

        :param force_save: create sync file every time in root directory
        :return: sync_dataframe
            example::

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
        out = ATLAS_CACHE_DIRECTORY / 'cellatlas_allen_sync.csv'

        if not out.exists() or force_save:
            ctlas = cls.load_from_csv()

            allen = load_structure_tree().select('name', 'acronym').sort('name')

            df = (
                ctlas.dataframe
                .rename({'Brain region': 'name'})
                .join(allen, on='name')
            )

            df.write_csv(out)
            fprint(f'SAVE cellatlas sync csv to {out}', vtype='io')

        return pl.read_csv(out)
