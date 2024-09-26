"""
Load example dataset from public google drive
"""
from __future__ import annotations

import shutil
from contextlib import contextmanager

import gdown
import numpy as np
import polars as pl

from neuralib.calimg.scanbox import SBXInfo
from neuralib.calimg.suite2p import Suite2PResult
from neuralib.io import NEUROLIB_CACHE_DIRECTORY
from neuralib.typing import PathLike

__all__ = [
    'load_example_scanbox',
    'load_example_dff',
    'load_example_suite2p',
    #
    'load_example_rois',
    #
    'load_ephys_meta',
    'load_ephys_data',
]


@contextmanager
def google_drive_file(file_id: str,
                      *,
                      quiet: bool = True,
                      output_dir: PathLike | None = None,
                      rename_file: str | None = None,
                      cached: bool = False,
                      invalid_cache: bool = False):
    if rename_file is not None:
        file = rename_file
    else:
        file = file_id

    if output_dir is None:
        output_dir = NEUROLIB_CACHE_DIRECTORY / 'tmp'

    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / file

    #
    if output_file.exists() and not invalid_cache:
        yield output_file
    else:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(output_file), quiet=quiet)

        try:
            yield output_file
        finally:
            if not cached:
                output_file.unlink()


@contextmanager
def google_drive_folder(folder_id: str,
                        *,
                        quiet: bool = True,
                        output_dir: PathLike | None = None,
                        rename_folder: str | None = None,
                        cached: bool = False,
                        invalid_cache: bool = False):
    if rename_folder is not None:
        folder_name = rename_folder
    else:
        folder_name = folder_id

    if output_dir is None:
        output_dir = NEUROLIB_CACHE_DIRECTORY / 'tmp' / folder_name

    #
    if output_dir.exists() and any(output_dir.iterdir()) and not invalid_cache:
        yield output_dir
    else:
        output_dir.mkdir(exist_ok=True, parents=True)
        gdown.download_folder(id=folder_id, output=str(output_dir), quiet=quiet)

        try:
            yield output_dir
        finally:
            if not cached:
                shutil.rmtree(output_dir)


# ==================== #
# Calcium Imaging data #
# ==================== #

def load_example_scanbox(**kwargs) -> SBXInfo:
    with google_drive_file('1Gcz_xRVWQJ9QMxq3vzZS8VruSbNiuh_s', **kwargs) as file:
        return SBXInfo.load(file)


def load_example_dff(**kwargs) -> np.ndarray:
    with google_drive_file('1OqGK2inSYkFEMEe_8umVr7VGz0fuSoX7', **kwargs) as file:
        return np.load(file, allow_pickle=True)


def load_example_suite2p(**kwargs) -> Suite2PResult:
    with google_drive_folder('1iVImr_rIywWhCiBDYhcphcSODaWJrhy7', **kwargs) as suite2p_dir:
        return Suite2PResult.load(suite2p_dir)


# ========== #
# ROIs Atlas #
# ========== #

def load_example_rois(**kwargs) -> pl.DataFrame:
    with google_drive_file('1cf2r3kcqjENBQMe8YzBQvol8tZgscN4J', **kwargs) as file:
        return pl.read_csv(file)


# ========== #
# Ephys data #
# ========== #

def load_ephys_meta(**kwargs):
    with google_drive_file('1gB2MegBAJbobEudx2pZmVe0LBQiU8HFU', **kwargs) as _:
        pass


def load_ephys_data(**kwargs):
    with google_drive_file('1U0xAchQagyXRT72M68fRQ4JsRQeW9q5d', **kwargs) as _:
        pass


if __name__ == '__main__':
    load_ephys_meta()
