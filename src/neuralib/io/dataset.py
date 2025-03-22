import pickle
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ContextManager

import gdown
import numpy as np
import polars as pl
from PIL import Image

from neuralib.imaging.scanbox import SBXInfo
from neuralib.imaging.suite2p import Suite2PResult
from neuralib.io import NEUROLIB_CACHE_DIRECTORY
from neuralib.tracking.deeplabcut.core import DeepLabCutResult, load_dlc_result
from neuralib.typing import PathLike

__all__ = [
    'google_drive_file',
    'google_drive_folder',
    #
    'load_example_rois',
    'load_example_rois_image',
    'load_example_dorsal_cortex',
    #
    'load_ephys_meta',
    'load_ephys_data',
    'load_npx2_reconstruction',
    #
    'load_example_scanbox',
    'load_example_suite2p',
    'load_example_rastermap_2p',
    'load_example_rastermap_wfield',
    #
    'load_example_dlc_h5',
    'load_example_dlc_csv'
]


@contextmanager
def google_drive_file(file_id: str,
                      *,
                      quiet: bool = True,
                      output_dir: PathLike | None = None,
                      rename_file: str | None = None,
                      cached: bool = False,
                      invalid_cache: bool = False) -> ContextManager[Path]:
    """
    Download file from Google Drive. If not ``cached``, then delete afterward.

    :param file_id: Google Drive file ID used to identify the file to be downloaded.
    :param quiet: Boolean flag to suppress output from the download process.
    :param output_dir: Directory path where the downloaded file will be saved.
    :param rename_file: Optional string to rename the downloaded file.
    :param cached: Boolean flag to retain the downloaded file after usage.
    :param invalid_cache: Boolean flag to force re-download even if the file exists in the cache.
    :return: A context manager yielding the path to the downloaded file.
    """
    if rename_file is not None:
        file = rename_file
    else:
        file = file_id

    if output_dir is None:
        output_dir = NEUROLIB_CACHE_DIRECTORY / 'tmp'

    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / file

    try:
        if output_file.exists() and not invalid_cache:
            yield output_file
        else:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(output_file), quiet=quiet)
            yield output_file
    finally:
        if not cached:
            output_file.unlink(missing_ok=True)


@contextmanager
def google_drive_folder(folder_id: str,
                        *,
                        quiet: bool = True,
                        output_dir: PathLike | None = None,
                        rename_folder: str | None = None,
                        cached: bool = False,
                        invalid_cache: bool = False) -> ContextManager[Path]:
    """
    Download a entire folder from Google Drive. If not ``cached``, then delete afterward.

    :param folder_id: Google Drive folder ID to download.
    :param quiet: If True, suppress the gdown output.
    :param output_dir: Directory where the folder should be downloaded. If None, a temporary directory is used.
    :param rename_folder: Optional new name for the downloaded folder.
    :param cached: If True, keep the downloaded folder for future use.
    :param invalid_cache: If True, force re-download even if cached data exists.
    :return: A context manager yielding the path to the downloaded folder.
    """
    if rename_folder is not None:
        folder_name = rename_folder
    else:
        folder_name = folder_id

    if output_dir is None:
        output_dir = NEUROLIB_CACHE_DIRECTORY / 'tmp' / folder_name

    try:
        if output_dir.exists() and any(output_dir.iterdir()) and not invalid_cache:
            yield output_dir
        else:
            output_dir.mkdir(exist_ok=True, parents=True)
            gdown.download_folder(id=folder_id, output=str(output_dir), quiet=quiet)
            yield output_dir
    finally:
        if not cached:
            shutil.rmtree(output_dir, ignore_errors=True)


# ========== #
# Atlas Data #
# ========== #

def load_example_rois(**kwargs) -> pl.DataFrame:
    """
    :param kwargs: Additional keyword arguments pass to ``google_drive_file`` to customize the loading behavior.
    :return: A Polars DataFrame containing the example ROIs data
    """
    with google_drive_file('1cf2r3kcqjENBQMe8YzBQvol8tZgscN4J', **kwargs) as file:
        return pl.read_csv(file)


def load_example_rois_image(**kwargs) -> np.ndarray:
    """
    :param kwargs: Additional arguments to be passed to the `google_drive_file` context manager.
    :return: An example imaging array with labeled ROIs
    """
    with google_drive_file('1-ZFC7Fd6IgwbY6X8oetvpGjZaBBiwg0t', **kwargs) as file:
        return np.array(Image.open(file))


def load_example_dorsal_cortex(color: bool = False, **kwargs) -> np.ndarray:
    """png file from the source svg

    .. seealso::

        :meth:`~neuralib.atlas.data.get_dorsal_cortex`

    :param color:

    """
    if color:
        file_id = '1Cujx3GGFZxEq0-isRlA_Ac-Puy6ml8IA'
    else:
        file_id = '1OEPpIIl8SszDJXO_1ADka-5pB556WNxc'

    with google_drive_file(file_id, **kwargs) as file:
        return np.array(Image.open(file))


# ========== #
# Ephys Data #
# ========== #

def load_ephys_meta(**kwargs):
    with google_drive_file('1gB2MegBAJbobEudx2pZmVe0LBQiU8HFU', **kwargs) as _:
        pass


def load_ephys_data(**kwargs):
    with google_drive_file('1U0xAchQagyXRT72M68fRQ4JsRQeW9q5d', **kwargs) as _:
        pass


def load_npx2_reconstruction(**kwargs) -> pl.DataFrame:
    """
    Example of NeuroPixel2 reconstruction data

    :param kwargs: Additional keyword arguments pass to ``google_drive_file`` to customize the loading behavior.
    :return: A Polars DataFrame containing the example DiI labelled traces ROIs
    """
    with google_drive_file('1fRvMNHhGgh5KP3CgGm6CMFth1qIAmwfh', **kwargs) as file:
        return pl.read_csv(file)


# ============ #
# Imaging Data #
# ============ #

def load_example_scanbox(**kwargs) -> SBXInfo:
    """
    :param kwargs: Additional keyword arguments pass to ``google_drive_file`` to customize the loading behavior.
    :return: An instance of ``SBXInfo`` loaded from the specified Google Drive file.
    """
    with google_drive_file('1Gcz_xRVWQJ9QMxq3vzZS8VruSbNiuh_s', **kwargs) as file:
        return SBXInfo.load(file)


def load_example_suite2p(**kwargs) -> Suite2PResult:
    """
    :param kwargs: Additional keyword arguments pass to ``google_drive_folder`` to customize the loading behavior.
    :return: An instance of ``Suite2PResult`` loaded with data from the specified Google Drive folder.
    """
    with google_drive_folder('1iVImr_rIywWhCiBDYhcphcSODaWJrhy7', **kwargs) as suite2p_dir:
        return Suite2PResult.load(suite2p_dir)


def load_example_rastermap_2p(**kwargs) -> dict[str, Any]:
    """
    :param kwargs: Additional arguments to be passed to the `google_drive_file` context manager.
    :return: A dictionary containing the 2-photon rastermap data cache
    """
    with google_drive_file('1SuzUhkbcdBY71dCnxjnT4pmQOsiOULBC', **kwargs) as file:
        with file.open('rb') as f:
            return pickle.load(f)


def load_example_rastermap_wfield(**kwargs) -> dict[str, Any]:
    """
    :param kwargs: Additional arguments to be passed to the `google_drive_file` context manager.
    :return: A dictionary containing the wide-field rastermap data cache
    """
    with google_drive_file('1zdZ3ihNPObyA1zVY7knJwVDH8MLXXuYB', **kwargs) as file:
        with file.open('rb') as f:
            return pickle.load(f)


# ========== #
# Behavioral #
# ========== #

def load_example_dlc_h5(**kwargs) -> DeepLabCutResult:
    with google_drive_file('1JNhx6Dpe8beP8vnh0yF3o3vY2DfUM-8A', rename_file='test_dlc.h5', **kwargs) as h5:
        with google_drive_file('1juICYcrXa7Vk-fQSBBSg2QcP9DGyHO2E', rename_file='test_dlc.pickle', **kwargs) as meta:
            return load_dlc_result(h5, meta)


def load_example_dlc_csv(**kwargs) -> DeepLabCutResult:
    with google_drive_file('1R2Ze5xjWlavcKvu45JOH3_QkOD4SSkVN', rename_file='test_dlc.csv', **kwargs) as csv:
        with google_drive_file('1juICYcrXa7Vk-fQSBBSg2QcP9DGyHO2E', rename_file='test_dlc.pickle', **kwargs) as meta:
            return load_dlc_result(csv, meta)
