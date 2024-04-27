from __future__ import annotations

from typing import List

import numpy as np
from typing_extensions import TypedDict

__all__ = ['KilosortProbe', 'Kilosort4Settings', 'KilosortPreprocessing', 'Kilosort4Options']


class KilosortProbe(TypedDict):
    chanMap: np.ndarray
    xc: np.ndarray
    yc: np.ndarray
    kcoords: np.ndarray
    n_chan: int


class Kilosort4Settings(TypedDict):
    # DEFAULT_SETTINGS
    n_chan_bin: int
    fs: float
    batch_size: int
    nblocks: int
    Th_universal: float
    Th_learned: float
    tmin: float
    tmax: float

    # EXTRA_PARAMETERS
    nt: int
    artifact_threshold: float
    nskip: int
    whitening_range: int
    binning_depth: float
    sig_interp: float
    nt0min: int
    dmin: float
    dminx: float
    min_template_size: float
    template_sizes: int
    nearest_chans: int
    nearest_templates: int
    templates_from_data: bool
    n_templates: int
    n_pcs: int
    Th_single_ch: float
    acg_threshold: float
    ccg_threshold: float
    cluster_downsampling: int
    cluster_pcs: int
    duplicate_spike_bins: int

    # Options
    do_CAR: bool
    invert_sign: bool
    do_correction: bool

    #
    filename: str
    data_dir: str
    settings: Kilosort4Settings
    probe: KilosortProbe
    data_dtype: str

    NTbuff: int
    Nchan: int
    torch_device: int

    results_dir: str


class KilosortPreprocessing(TypedDict, total=False):
    whiten_mat: np.ndarray
    hp_filter: np.ndarray


class Kilosort4Options(TypedDict, Kilosort4Settings, KilosortProbe):
    Nbatches: int
    preprocessing: KilosortPreprocessing

    Wrot: np.ndarray
    fwav: np.ndarray
    wPCA: np.ndarray
    wTEMP: np.ndarray
    yup: np.ndarray
    xup: np.ndarray
    ycup: np.ndarray
    xcup: np.ndarray
    iC: np.ndarray
    iC2: np.ndarray
    weigh: np.ndarray
    yblk: np.ndarray
    dshift: np.ndarray
    iKxx: np.ndarray
    iCC: np.ndarray
    iU: np.ndarray
    runtime: float
    is_tensor: List[str]
