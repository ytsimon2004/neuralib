from __future__ import annotations

from pathlib import Path
from typing import Literal, overload

import numpy as np
import polars as pl
from brainglobe_atlasapi import BrainGlobeAtlas
from tqdm import tqdm

from neuralib.io import save_json, load_json
from neuralib.io.core import ATLAS_CACHE_DIRECTORY
from neuralib.typing import PathLike
from neuralib.util.tqdm import download_with_tqdm
from neuralib.util.verbose import print_save, print_load

__all__ = [
    'ATLAS_NAME',
    'load_bg_structure_tree',
    'load_bg_volumes',
    'get_children',
    'get_annotation_ids',
    'get_leaf_in_annotation',
    'build_annotation_leaf_map',
    #
    'get_dorsal_cortex'
]

ATLAS_NAME = Literal[
    'allen_mouse_10um',
    'allen_mouse_25um',
    'allen_mouse_50um',
    'allen_mouse_100um',
    'kim_mouse_10um',
    'kim_mouse_25um',
    'kim_mouse_50um',
    'kim_mouse_100um',
    'perens_lsfm_mouse_20um',
    'perens_stereotaxic_mouse_mri_25um',
    'princeton_mouse_20um',
]
"""Atlas Name From BrainGlobeAtlas"""


def load_bg_structure_tree(atlas_name: ATLAS_NAME = 'allen_mouse_10um', *,
                           check_latest: bool = True,
                           paired: bool = False) -> pl.DataFrame:
    """
    Load structure dataframe or dict from `brainglobe_atlasapi`

    :param atlas_name: :attr:`~neuralib.atlas.data.ATLAS_NAME`
    :param check_latest: If check the brainglobe api latest version
    :param paired: To only ``acronym`` & ``parent_acronym`` fields
    :return:
    """
    file = BrainGlobeAtlas(atlas_name, check_latest=check_latest).root_dir / 'structures.csv'
    df = pl.read_csv(file).with_columns(pl.col('parent_structure_id').cast(pl.Int64))
    df = df.join(
        df.select([pl.col("id").alias("parent_structure_id"), pl.col("acronym").alias("parent_acronym")]),
        on="parent_structure_id",
        how="left"
    )

    if paired:
        name = df.select(pl.col('acronym'), pl.col('id'), pl.col('parent_structure_id'))
        join_df = name.join(name, left_on='parent_structure_id', right_on='id')
        parent_child = join_df.select(pl.col('acronym'), pl.col('acronym_right').alias('parent_acronym'))
        return parent_child
    else:
        return df


def load_bg_volumes(atlas_name: ATLAS_NAME = 'allen_mouse_10um',
                    cached_file: PathLike | None = None,
                    force: bool = False) -> pl.DataFrame:
    """
    Load structure tree dataframe with volume for each region ``volume_mm3`` ::

        ┌─────────┬─────┬─────────────────┬─────────────────┬────────────────┬────────────────┬────────────┐
        │ acronym ┆ id  ┆ name            ┆ structure_id_pa ┆ parent_structu ┆ parent_acronym ┆ volume_mm3 │
        │ ---     ┆ --- ┆ ---             ┆ th              ┆ re_id          ┆ ---            ┆ ---        │
        │ str     ┆ i64 ┆ str             ┆ ---             ┆ ---            ┆ str            ┆ f64        │
        │         ┆     ┆                 ┆ str             ┆ i64            ┆                ┆            │
        ╞═════════╪═════╪═════════════════╪═════════════════╪════════════════╪════════════════╪════════════╡
        │ VI      ┆ 653 ┆ Abducens        ┆ /997/8/343/1065 ┆ 370            ┆ MY-mot         ┆ 0.030332   │
        │         ┆     ┆ nucleus         ┆ /354/370/653/   ┆                ┆                ┆            │
        │ AOB     ┆ 151 ┆ Accessory       ┆ /997/8/567/688/ ┆ 698            ┆ OLF            ┆ 0.652032   │
        │         ┆     ┆ olfactory bulb  ┆ 695/698/151/    ┆                ┆                ┆            │
        │ …       ┆ …   ┆ …               ┆ …               ┆ …              ┆ …              ┆ …          │
        │ von     ┆ 949 ┆ vomeronasal     ┆ /997/1009/967/9 ┆ 967            ┆ cm             ┆ 0.013428   │
        │         ┆     ┆ nerve           ┆ 49/             ┆                ┆                ┆            │
        └─────────┴─────┴─────────────────┴─────────────────┴────────────────┴────────────────┴────────────┘

    :param atlas_name: :attr:`~neuralib.atlas.data.ATLAS_NAME`
    :param cached_file: Cached file path.
    :param force: Force overwrite the cached file
    :return:
    """
    if cached_file is None:
        cached_file = ATLAS_CACHE_DIRECTORY / f'{atlas_name}_bg_volumes.csv'

    if cached_file.exists() and not force:
        print_load(cached_file)
        return pl.read_csv(cached_file)
    else:
        df = load_bg_structure_tree(atlas_name)
        lut = build_annotation_leaf_map(atlas_name)
        bg = BrainGlobeAtlas(atlas_name)
        flat_annotation = bg.annotation.ravel()
        voxel_volume_mm3 = (bg.resolution[0] / 1000) ** 3

        volumes = []
        ids = df['id'].to_list()

        for region_id in tqdm(ids, desc=f"calculating volumes and save cache ({atlas_name})"):
            leaf_ids = lut.get(region_id, [])

            if not leaf_ids:
                volumes.append(-1)  # if not found
                continue

            mask = np.isin(flat_annotation, leaf_ids)
            count = np.count_nonzero(mask)
            vol_mm3 = count * voxel_volume_mm3
            volumes.append(vol_mm3)

        ret = df.with_columns(pl.Series(name='volume_mm3', values=volumes))
        ret.write_csv(cached_file)
        print_save(cached_file)

        return ret


@overload
def get_children(parent: int, *,
                 dataframe: bool = False,
                 atlas_name: ATLAS_NAME = 'allen_mouse_10um') -> list[int] | pl.DataFrame:
    pass


@overload
def get_children(parent: str, *,
                 dataframe: bool = False,
                 atlas_name: ATLAS_NAME = 'allen_mouse_10um') -> list[str] | pl.DataFrame:
    pass


def get_children(parent: int | str, *,
                 dataframe: bool = False,
                 atlas_name: ATLAS_NAME = 'allen_mouse_10um') -> list[str] | pl.DataFrame:
    """
    Get children brain region id or acronym from its parent

    :param parent: id or acronym
    :param dataframe: return as dataframe, otherwise return as list
    :param atlas_name: :attr:`~neuralib.atlas.data.ATLAS_NAME`
    :return:
    """
    df = load_bg_structure_tree(atlas_name=atlas_name)
    return _get_children(df, parent, dataframe)


def _get_children(df, parent, dataframe):
    if isinstance(parent, int):
        ret = df.filter(pl.col('parent_structure_id') == parent)
        field = 'id'
    elif isinstance(parent, str):
        ret = df.filter(pl.col('parent_acronym') == parent)
        field = 'acronym'
    else:
        raise TypeError('')

    if not dataframe:
        ret = ret[field].to_list()

    return ret


# ============= #
# BG Annotation #
# ============= #

def get_annotation_ids(atlas_name: ATLAS_NAME = 'allen_mouse_10um', check_latest: bool = True) -> np.ndarray:
    """
    Get unique annotation id

    :param atlas_name: :attr:`~neuralib.atlas.data.ATLAS_NAME`
    :param check_latest:
    :return:
    """
    annotation = BrainGlobeAtlas(atlas_name, check_latest=check_latest).annotation
    return np.unique(annotation)


def get_leaf_in_annotation(region: int | str, *,
                           name: bool = False,
                           cached_file: PathLike | None = None,
                           atlas_name: ATLAS_NAME = 'allen_mouse_10um') -> list[int] | list[str]:
    """
    Get a list of annotation {id, acronym} with given region {id, acronym}

    :param region: Region id or region acronym
    :param name: If True, return acronym, otherwise return id
    :param cached_file: Cached json for the annotation_leaf_map
    :param atlas_name: :attr:`~neuralib.atlas.data.ATLAS_NAME`
    :return: List of annotation {id, acronym}
    """
    tree = load_bg_structure_tree(atlas_name=atlas_name)

    # to id
    if isinstance(region, str):
        region_ids = tree.filter(pl.col('acronym') == region)['id'].to_list()
        if len(region_ids) != 1:
            raise RuntimeError(f"The region {region} is not a valid acronym")
        region = region_ids[0]

    dy = build_annotation_leaf_map(cached_file=cached_file)

    try:
        result = dy[region]
    except KeyError:
        raise ValueError(f'Invalid region: {region}')

    if name:
        result = tree.filter(pl.col('id').is_in(result))['acronym'].to_list()

    return result


def build_annotation_leaf_map(atlas_name: ATLAS_NAME = 'allen_mouse_10um', *,
                              cached_file: PathLike | None = None,
                              force: bool = False) -> dict[int, list[int]]:
    """
    Get all region id (key) and list of annotation id (values)

    :param atlas_name: :attr:`~neuralib.atlas.data.ATLAS_NAME`
    :param cached_file: Cached json file path
    :param force: Force re-compute the cached file
    :return:
    """
    if cached_file is None:
        cached_file = ATLAS_CACHE_DIRECTORY / f'{atlas_name}_annotation_leaf.json'

    if Path(cached_file).suffix != '.json':
        raise ValueError('not a json file')

    #
    if cached_file.exists() and not force:
        data = load_json(cached_file, verbose=False)
        leaf_map = {int(k): v for k, v in data.items()}
    else:
        tree = load_bg_structure_tree(atlas_name)
        id_to_children = _build_id_to_children_map(tree)
        annotation_ids = set(get_annotation_ids())

        leaf_map = {}

        def collect(rid):
            if rid in leaf_map:
                return leaf_map[rid]
            if rid in annotation_ids:
                leaf_map[rid] = [rid]
            else:
                result = []
                for child in id_to_children.get(rid, []):
                    result.extend(collect(child))
                leaf_map[rid] = result
            return leaf_map[rid]

        all_ids = tree['id'].to_list()
        for rid in all_ids:
            collect(rid)

        save_json(cached_file, leaf_map)

    return leaf_map


def _build_id_to_children_map(tree: pl.DataFrame) -> dict[int, list[int]]:
    df = tree.select(['id', 'parent_structure_id'])
    grouped = df.group_by('parent_structure_id', maintain_order=False).agg(pl.col('id'))
    return {row['parent_structure_id']: row['id'] for row in grouped.iter_rows(named=True)}


# =============== #
# Allen Resources #
# =============== #

def get_dorsal_cortex(output_dir: Path | None = None) -> Path:
    """
    Get example dorsal projection annotation svg file

    .. seealso::

        https://community.brain-map.org/t/aligning-dorsal-projection-of-mouse-common-coordinate-framework-with-wide-field-images-of-mouse-brain/140/2

    :param output_dir: Output directory for caching
    :return: Output file path
    """

    if output_dir is None:
        output_dir = ATLAS_CACHE_DIRECTORY

    filename = 'cortical_map_top_down.svg'
    output = output_dir / filename

    if not output.exists():
        url = 'http://connectivity.brain-map.org/assets/cortical_map_top_down.svg'
        content = download_with_tqdm(url)

        with open(output, 'wb') as f:
            f.write(content.getvalue())
            print_save(output, verb='DOWNLOAD')

    return output
