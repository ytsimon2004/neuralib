from __future__ import annotations

import warnings
from typing import Literal

import nrrd
import numpy as np

from neuralib.util.io import CCF_CACHE_DIRECTORY, ALLEN_SDK_DIRECTORY
from neuralib.util.tqdm import download_with_tqdm
from neuralib.util.util_type import PathLike
from neuralib.util.util_verbose import fprint

__all__ = [
    'load_ccf_annotation',
    'load_ccf_template',
    'load_allensdk_annotation'
]

DATA_SOURCE_TYPE = Literal['ccf_annotation', 'ccf_template', 'allensdk_annotation']


def _cache_nparray(url: str, file: PathLike):
    fprint(f'DOWNLOADING... {file.name} from {url}', vtype='io')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        content = download_with_tqdm(url)
        data = np.load(content, allow_pickle=True)
        np.save(file, data)


def load_ccf_annotation(output_dir: PathLike | None = None) -> np.ndarray:
    """
    Annotation volume file from AllenCCF pipeline

    .. seealso::

        https://github.com/cortex-lab/allenCCF

    :param output_dir: output directory for caching
    :return:
    """
    if output_dir is None:
        output_dir = CCF_CACHE_DIRECTORY
        if not CCF_CACHE_DIRECTORY.exists():
            CCF_CACHE_DIRECTORY.mkdir(exist_ok=True, parents=True)

    file = output_dir / 'annotation_volume_10um_by_index.npy'

    if not file.exists():
        url = 'https://figshare.com/ndownloader/files/44925493'
        _cache_nparray(url, file)

    return np.load(file, allow_pickle=True)


def load_ccf_template(output_dir: PathLike | None = None) -> np.ndarray:
    """
    Template volume file from AllenCCF pipeline

    .. seealso::

        https://github.com/cortex-lab/allenCCF

    :param output_dir: output directory for caching
    :return:
    """
    if output_dir is None:
        output_dir = CCF_CACHE_DIRECTORY
        if not CCF_CACHE_DIRECTORY.exists():
            CCF_CACHE_DIRECTORY.mkdir(exist_ok=True, parents=True)

    file = output_dir / 'template_volume_10um.npy'

    if not file.exists():
        url = 'https://figshare.com/ndownloader/files/44925496'
        _cache_nparray(url, file)

    return np.load(file, allow_pickle=True)


def load_allensdk_annotation(resolution: int = 10,
                             output_dir: PathLike | None = None) -> np.ndarray:
    """
    Data Source directly from Allen Institute

    .. seealso::

        https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/

    :param resolution: volume resolution in um. default is 10 um
    :param output_dir: output directory for caching
    :return:
    """
    if output_dir is None:
        output_dir = ALLEN_SDK_DIRECTORY
        if not ALLEN_SDK_DIRECTORY.exists():
            ALLEN_SDK_DIRECTORY.mkdir(exist_ok=True, parents=True)

    file = output_dir / f'annotation_{resolution}.nrrd'

    if not file.exists():
        from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
        mcapi = MouseConnectivityApi()
        version = MouseConnectivityApi.CCF_VERSION_DEFAULT

        mcapi.download_annotation_volume(version, resolution, file)

    return nrrd.read(file)[0]
