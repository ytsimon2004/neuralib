import dataclasses
import io
from pathlib import Path

import pandas as pd
import polars as pl

from neuralib.atlas._deprecate import _CellAtlas
from neuralib.atlas.data import load_bg_structure_tree
from neuralib.io.core import ATLAS_CACHE_DIRECTORY
from neuralib.typing import PathLike
from neuralib.util.deprecation import deprecated_class
from neuralib.util.utils import ensure_dir
from neuralib.util.verbose import print_save

__all__ = ['load_cellatlas', 'CellAtlas']


def load_cellatlas(file: PathLike | None = None, *,
                   with_cell_type: bool = False,
                   with_detail: bool = False,
                   with_total_neurons: bool = True,
                   with_acronym: bool = True) -> pl.DataFrame:
    """
    Load the dataframe with cell types and volume information for each brain area

    .. seealso::

        `Rodarie D et al., (2022) <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010739#sec047>`_

    :param file: Cached csv filepath. If not exist, download from the source paper
    :param with_cell_type: With cell type information, defaults to False
    :param with_detail:  With some outlier brain areas, defaults to False
    :param with_total_neurons: With ``n_neurons`` field, defaults to True
    :param with_acronym: With ``acronym`` field sync with structure tree data, defaults to True
    :return: DataFrame
    """
    if file is None:
        file = ensure_dir(ATLAS_CACHE_DIRECTORY) / 'cellatlas.csv'

    if not Path(file).exists():
        df = _request(file)
    else:
        df = pl.read_csv(file).rename({'Brain region': 'name'})

    if not with_cell_type:
        df = df.select('name', 'Neuron [mm^-3]', 'Volumes [mm^3]')

    if not with_detail:
        patterns = (',', '/', r'\(')
        for pt in patterns:
            df = df.filter(~(pl.col('name').str.contains(pt)))

    if with_total_neurons:
        expr = (pl.col('Neuron [mm^-3]') * pl.col('Volumes [mm^3]')).alias('n_neurons').cast(pl.Int64)
        df = df.with_columns(expr).drop('Neuron [mm^-3]')

    if with_acronym:
        tree = load_bg_structure_tree().select('name', 'acronym').sort('name')
        df = df.join(tree, on='name')

    return df


def _request(output: Path) -> pl.DataFrame:
    """download from paper source"""
    import requests

    url = 'https://journals.plos.org/ploscompbiol/article/file?type=supplementary&id=10.1371/journal.pcbi.1010739.s011'
    resp = requests.get(url)

    if resp.status_code == 200:
        df = pd.read_excel(io.BytesIO(resp.content), sheet_name='Densities BBCAv1')
        pl.from_pandas(df).write_csv(output)
        print_save(output, verb='DOWNLOAD')
    else:
        raise RuntimeError('download cellatlas FAIL')

    return df


@deprecated_class(new='load_cellatlas()', removal_version='v0.5.0')
@dataclasses.dataclass
class CellAtlas(_CellAtlas):
    ...
