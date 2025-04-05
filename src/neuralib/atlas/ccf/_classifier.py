import shutil
from pathlib import Path
from typing import TypedDict

import polars as pl

from neuralib.atlas.ccf.core import (
    AbstractCCFDir,
    SagittalCCFDir,
    SagittalCCFOverlapDir,
    CoronalCCFOverlapDir,
    CoronalCCFDir
)
from neuralib.atlas.typing import Area, HEMISPHERE_TYPE, Source, Channel
from neuralib.atlas.util import PLANE_TYPE
from neuralib.util.utils import uglob

__all__ = [
    'FluorReprType',
    #
    'UserInjectionConfig',
    #
    #
    '_concat_channel',

]

FluorReprType = dict[Channel, Source]


# ======= #
# Configs #
# ======= #

class UserInjectionConfig(TypedDict):
    area: Area
    """injection area"""
    hemisphere: HEMISPHERE_TYPE
    """injection hemisphere"""
    ignore: bool
    """whether local roi counts will be ignored"""  # TODO if needed?
    fluor_repr: FluorReprType
    """fluorescence color and tracing source alias pairs"""


# Example (replace to user-specific) TODO as doc
_DEFAULT_RSP_CONFIG = UserInjectionConfig(
    area='RSP',
    hemisphere='ipsi',
    ignore=True,
    fluor_repr=dict(
        rfp='pRSC',
        gfp='aRSC',
        overlap='overlap'
    )
)


# noinspection PyTypeChecker
def _concat_channel(ccf_dir: AbstractCCFDir, plane: PLANE_TYPE) -> pl.DataFrame:
    """
    Find the csv data from `labelled_roi_folder`, if multiple files are found, concat to single df.
    `channel` & `source` columns are added to the dataframe.
    
    If sagittal slice, auto move ipsi/contra hemispheres dataset (`resize_ipsi`, `resize_contra`) 
    to new `resize` directory

    :param ccf_dir: :class:`~neuralib.atlas.ccf.core.AbstractCCFDir()`
    :param plane: ``PLANE_TYPE`` {'coronal', 'sagittal', 'transverse'}
    :return:
    """
    if plane == 'sagittal':
        _auto_sagittal_combine(ccf_dir)
    elif plane == 'coronal':
        _auto_coronal_combine(ccf_dir)


def _auto_overlap_copy(ccf: CoronalCCFOverlapDir | SagittalCCFOverlapDir) -> None:
    src = uglob(ccf.labelled_roi_folder_overlap, '*.csv')
    filename = f'{ccf.animal}_overlap_roitable'
    if ccf.plane_type == 'sagittal':
        filename += f'_{ccf.hemisphere}'

    dst = (ccf.labelled_roi_folder / filename).with_suffix('.csv')
    shutil.copy(src, dst)
    print(f'copy overlap file from {src} to {dst}')


def _auto_coronal_combine(ccf_dir: CoronalCCFDir | CoronalCCFOverlapDir):
    _auto_overlap_copy(ccf_dir)


def _auto_sagittal_combine(ccf_dir: SagittalCCFDir | SagittalCCFOverlapDir) -> None:
    """copy file from overlap dir to major fluorescence (channel) folder,
    then combine different hemisphere data"""

    old_args = ccf_dir.hemisphere

    def with_hemisphere_stem(ccf: SagittalCCFDir | SagittalCCFOverlapDir) -> list[Path]:
        ls = list(ccf.labelled_roi_folder.glob('*.csv'))
        for it in ls:
            if ccf.hemisphere not in it.name:
                new_path = it.with_stem(it.stem + f'_{ccf.hemisphere}')
                it.rename(new_path)

        return list(ccf.labelled_roi_folder.glob('*.csv'))  # new glob

    mv_list = []

    ccf_dir.hemisphere = 'ipsi'
    if isinstance(ccf_dir, SagittalCCFOverlapDir):
        _auto_overlap_copy(ccf_dir)
    ext = with_hemisphere_stem(ccf_dir)
    mv_list.extend(ext)

    #
    ccf_dir.hemisphere = 'contra'
    if isinstance(ccf_dir, SagittalCCFOverlapDir):
        _auto_overlap_copy(ccf_dir)
    ext = with_hemisphere_stem(ccf_dir)
    mv_list.extend(ext)

    #
    ccf_dir.hemisphere = 'both'  # as resize
    target = ccf_dir.labelled_roi_folder
    for file in mv_list:
        shutil.copy(file, target / file.name)
        print(f'copy file from {file} to {target / file.name}')

    ccf_dir.hemisphere = old_args  # assign back
