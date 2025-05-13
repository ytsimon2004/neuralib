import pickle
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ContextManager, TYPE_CHECKING

import gdown
import numpy as np
import polars as pl
from PIL import Image

from neuralib.io import NEUROLIB_DATASET_DIRECTORY
from neuralib.typing import PathLike

if TYPE_CHECKING:
    from neuralib.tracking.deeplabcut import DeepLabCutDataFrame
    from neuralib.imaging.suite2p import Suite2PResult
    from neuralib.imaging.scanbox import SBXInfo
    from neuralib.tracking.facemap import FaceMapResult
    from neuralib.model.rastermap import RasterMapResult
    from neuralib.scan.czi import CziScanner
    from neuralib.scan.lsm import TiffScanner

__all__ = [
    'google_drive_file',
    'google_drive_folder',
    'clean_all_cache_dataset',
    #
    'load_example_rois',
    'load_example_rois_image',
    'load_example_rois_dir',
    'load_example_dorsal_cortex',
    #
    'load_ephys_meta',
    'load_ephys_data',
    'load_npx2_reconstruction',
    #
    'load_example_scanbox',
    'load_example_suite2p_result',
    'load_example_rastermap_2p_result',
    'load_example_rastermap_2p_cache',
    'load_example_retinotopic_data',
    #
    'load_example_dlc_h5',
    'load_example_dlc_csv',
    'load_example_facemap_pupil',
    'load_example_facemap_keypoints',
    #
    'load_example_lsm',
    'load_example_czi'
]


@contextmanager
def google_drive_file(file_id: str,
                      *,
                      quiet: bool = False,
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
        output_dir = NEUROLIB_DATASET_DIRECTORY

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
                        quiet: bool = False,
                        output_dir: Path | None = None,
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
        output_dir = NEUROLIB_DATASET_DIRECTORY / folder_name

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


def clean_all_cache_dataset():
    """removes the cache dataset directory"""
    shutil.rmtree(NEUROLIB_DATASET_DIRECTORY)
    print(f'[REMOVE]: {NEUROLIB_DATASET_DIRECTORY}')


# ========== #
# Atlas Data #
# ========== #

def load_example_rois(**kwargs) -> pl.DataFrame:
    """
    Load example ROIs dataframe

    :param kwargs: Additional keyword arguments pass to ``google_drive_file`` to customize the loading behavior.
    :return: A Polars DataFrame containing the example ROIs data
    """
    with google_drive_file('1dKpZt6eF4szvl7svWRdBQkOfTVLQi4Xg', **kwargs) as file:
        return pl.read_csv(file)


def load_example_rois_image(**kwargs) -> np.ndarray:
    """
    Load example ROIs image array

    :param kwargs: Additional arguments to be passed to the `google_drive_file` context manager.
    :return: An example imaging array with labeled ROIs
    """
    with google_drive_file('1-ZFC7Fd6IgwbY6X8oetvpGjZaBBiwg0t', **kwargs) as file:
        return np.array(Image.open(file))


def load_example_rois_dir(**kwargs) -> Path:
    """Load a directory containing multiple example ROIs image array"""
    with google_drive_folder('1tj36-lzOArjwMsZlhbyy_B19ffl50J98', **kwargs) as d:
        return d


def load_example_dorsal_cortex(color: bool = False, **kwargs) -> np.ndarray:
    """Load dorsal cortex image array from the source svg

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

def load_example_scanbox(**kwargs) -> 'SBXInfo':
    """
    Load example ScanBox data

    :param kwargs: Additional keyword arguments pass to ``google_drive_file`` to customize the loading behavior.
    :return: An instance of ``SBXInfo`` loaded from the specified Google Drive file.
    """
    from neuralib.imaging.scanbox import SBXInfo
    with google_drive_file('1Gcz_xRVWQJ9QMxq3vzZS8VruSbNiuh_s', **kwargs) as file:
        return SBXInfo.load(file)


def load_example_suite2p_result(**kwargs) -> 'Suite2PResult':
    """
    Load example Suite2P data

    :param kwargs: Additional keyword arguments pass to ``google_drive_folder`` to customize the loading behavior.
    :return: An instance of ``Suite2PResult`` loaded with data from the specified Google Drive folder.
    """
    from neuralib.imaging.suite2p import read_suite2p
    with google_drive_folder('1iVImr_rIywWhCiBDYhcphcSODaWJrhy7', **kwargs) as suite2p_dir:
        return read_suite2p(suite2p_dir)


def load_example_rastermap_2p_result(**kwargs) -> 'RasterMapResult':
    """Load example rastermap 2P data"""
    from neuralib.model.rastermap import read_rastermap
    with google_drive_file('1KSic4sXyF3hTgQbGijMpa3D3TGJUU097', **kwargs) as file:
        return read_rastermap(file)


def load_example_rastermap_2p_cache(**kwargs) -> dict[str, Any]:
    """
    Load example rastermap 2P cache pickle file
    :param kwargs: Additional arguments to be passed to the `google_drive_file` context manager.
    :return: A dictionary containing the 2-photon rastermap data cache
    """
    with google_drive_file('1SuzUhkbcdBY71dCnxjnT4pmQOsiOULBC', **kwargs) as file:
        with file.open('rb') as f:
            return pickle.load(f)


def load_example_retinotopic_data(**kwargs) -> Path:
    """Load example retinotopic tiff file path"""
    with google_drive_file('1J8iqP_EBaknNJehRUw3nwp7lEw2_UFXz',
                           quiet=True,
                           cached=True,
                           rename_file='retinotopic.tiff', **kwargs) as file:
        return Path(file)


# ========== #
# Behavioral #
# ========== #

def load_example_dlc_h5(**kwargs) -> 'DeepLabCutDataFrame':
    """Load example Deeplabcut h5"""
    with google_drive_file('1JNhx6Dpe8beP8vnh0yF3o3vY2DfUM-8A', rename_file='test_dlc.h5', **kwargs) as h5:
        from neuralib.tracking.deeplabcut.core import read_dlc
        with google_drive_file('1juICYcrXa7Vk-fQSBBSg2QcP9DGyHO2E', rename_file='test_dlc.pickle', **kwargs) as meta:
            return read_dlc(h5, meta)


def load_example_dlc_csv(**kwargs) -> 'DeepLabCutDataFrame':
    """Load example Deeplabcut csv"""
    from neuralib.tracking.deeplabcut.core import read_dlc
    with google_drive_file('1R2Ze5xjWlavcKvu45JOH3_QkOD4SSkVN', rename_file='test_dlc.csv', **kwargs) as csv:
        with google_drive_file('1juICYcrXa7Vk-fQSBBSg2QcP9DGyHO2E', rename_file='test_dlc.pickle', **kwargs) as meta:
            return read_dlc(csv, meta)


def load_example_facemap_pupil(**kwargs) -> 'FaceMapResult':
    """Load example facemap pupil data"""
    from neuralib.tracking.facemap import read_facemap
    with google_drive_folder('1cacH5DWLmYqh_7PLwqEasmER_TfKgZ1b', **kwargs) as pupil_dir:
        return read_facemap(pupil_dir)


def load_example_facemap_keypoints(**kwargs) -> 'FaceMapResult':
    """Load example facemap keypoint data"""
    from neuralib.tracking.facemap import read_facemap
    with google_drive_folder('1FWz70HE_hQuhE6K9hoO_y1OgeG11NsGM', **kwargs) as pupil_dir:
        return read_facemap(pupil_dir)


# ============= #
# Confocal Scan #
# ============= #

def load_example_lsm(**kwargs) -> 'TiffScanner':
    """load example lsm file"""
    from neuralib.scan.lsm import lsm_file
    with google_drive_file('1beq6PCY8XmZjyWiOk2-KkcXLrmbLpWmS', rename_file='test.lsm', **kwargs) as file:
        with lsm_file(file) as lsm:
            return lsm


def load_example_czi(**kwargs) -> 'CziScanner':
    """load example czi file"""
    from neuralib.scan.czi import czi_file
    with google_drive_file('1gSPz_a7kCZ3UQABC-v-_sNnAxhfY_ly0', rename_file='test.czi', **kwargs) as file:
        with czi_file(file) as czi:
            return czi
