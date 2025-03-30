from pathlib import Path
from typing import Literal, overload

import nrrd
import numpy as np
import polars as pl
from brainglobe_atlasapi import BrainGlobeAtlas

from neuralib.io import save_json, load_json
from neuralib.io.core import ALLEN_SDK_DIRECTORY, ATLAS_CACHE_DIRECTORY
from neuralib.typing import PathLike
from neuralib.util.deprecation import deprecated_func
from neuralib.util.tqdm import download_with_tqdm
from neuralib.util.verbose import fprint, print_save

__all__ = [
    'load_bg_structure_tree',
    'get_children',
    'get_annotation_ids',
    'get_leaf_in_annotation',
    'build_annotation_leaf_map',
    #
    'get_dorsal_cortex',
    #
    'load_allensdk_annotation',
    'load_ccf_annotation',
    'load_ccf_template',
    'load_structure_tree',

]


def load_bg_structure_tree(atlas_name: str = 'allen_mouse_10um',
                           check_latest: bool = True,
                           paired: bool = False) -> pl.DataFrame:
    """
    Load structure dataframe or dict from `brainglobe_atlasapi`

    :param atlas_name: allen source name
    :param check_latest: if check the brainglobe api latest version
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
        parent_child = join_df.select(pl.col('acronym'), pl.col('names_right').alias('parent_acronym'))

        return parent_child
    else:
        return df


@overload
def get_children(parent: int, *, dataframe: bool = False, atlas_name: str = 'allen_mouse_10um') -> list[int] | pl.DataFrame:
    pass


@overload
def get_children(parent: str, *, dataframe: bool = False, atlas_name: str = 'allen_mouse_10um') -> list[str] | pl.DataFrame:
    pass


def get_children(parent: int | str, dataframe: bool = False, atlas_name: str = 'allen_mouse_10um') -> list[str] | pl.DataFrame:
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

def get_annotation_ids(atlas_name: str = 'allen_mouse_10um', check_latest: bool = True) -> np.ndarray:
    annotation = BrainGlobeAtlas(atlas_name, check_latest=check_latest).annotation
    return np.unique(annotation)


def get_leaf_in_annotation(region: int | str, *,
                           name: bool = False,
                           cached_file: PathLike | None = None) -> list[int] | list[str]:
    """
    Get a list of annotation {id, acronym} with given region {id, acronym}

    :param region: region id or region acronym
    :param name: If True, return acronym, otherwise return id
    :param cached_file: cached json for the annotation_leaf_map
    :return: List of annotation {id, acronym}
    """
    tree = load_bg_structure_tree()

    # to id
    if isinstance(region, str):
        region_ids = tree.filter(pl.col('acronym') == region)['id'].to_list()
        if len(region_ids) != 1:
            raise RuntimeError(f"The region {region} is not a valid acronym")
        region = region_ids[0]

    dy = build_annotation_leaf_map(cached_file)

    try:
        result = dy[region]
    except KeyError:
        raise ValueError(f'Invalid region: {region}')

    if name:
        result = tree.filter(pl.col('id').is_in(result))['acronym'].to_list()

    return result


def build_annotation_leaf_map(cached_file: PathLike | None = None) -> dict[int, list[int]]:
    """
    Get all region id (key) and list of annotation id (values)

    :param cached_file: cached json file
    :return:
    """
    if cached_file is None:
        cached_file = ATLAS_CACHE_DIRECTORY / 'annotation_leaf.json'

    if Path(cached_file).suffix != '.json':
        raise ValueError('not a json file')

    #
    if cached_file.exists():
        data = load_json(cached_file, verbose=False)
        leaf_map = {int(k): v for k, v in data.items()}
    else:
        tree = load_bg_structure_tree()
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


# ================ #
# TO BE DEPRECATED #
# ================ #

@deprecated_func(removal_version='0.4.3', remarks='switch brainglobe api instead')
def load_ccf_annotation(output_dir: PathLike | None = None) -> np.ndarray:
    from ._deprecate import _load_ccf_annotation
    return _load_ccf_annotation(output_dir)


@deprecated_func(removal_version='0.4.3', remarks='switch brainglobe api instead')
def load_ccf_template(output_dir: PathLike | None = None) -> np.ndarray:
    from ._deprecate import _load_ccf_template
    return _load_ccf_template(output_dir)


@deprecated_func(removal_version='0.4.3', remarks='switch brainglobe api instead')
def load_structure_tree(version: Literal['2017', 'old'] = '2017', output_dir: PathLike | None = None) -> pl.DataFrame:
    from ._deprecate import _load_structure_tree
    return _load_structure_tree(version, output_dir)


@deprecated_func(removal_version='0.4.3', remarks='switch brainglobe api instead, and probably deprecate allensdk dependency')
def load_allensdk_annotation(resolution: int = 10, output_dir: PathLike | None = None) -> np.ndarray:
    """
    Data Source directly from Allen Institute

    .. seealso::

        https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/

    :param resolution: volume resolution in um. default is 10 um
    :param output_dir: output directory for caching
    :return: Array[uint32, [AP, DV, ML]]
    """
    if output_dir is None:
        output_dir = ALLEN_SDK_DIRECTORY
        if not ALLEN_SDK_DIRECTORY.exists():
            ALLEN_SDK_DIRECTORY.mkdir(exist_ok=True, parents=True)

    file = output_dir / f'annotation_{resolution}.nrrd'

    if not file.exists():
        try:
            from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
        except ImportError as e:
            fprint('Build error from project.toml. Please manually install using "pip install allensdk --no-deps"', vtype='error')
            raise e

        mcapi = MouseConnectivityApi()
        version = MouseConnectivityApi.CCF_VERSION_DEFAULT

        mcapi.download_annotation_volume(version, resolution, file)

    return nrrd.read(file)[0]
