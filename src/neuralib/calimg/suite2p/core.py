from __future__ import annotations

import datetime
from pathlib import Path
from typing import Literal, TypedDict, final

import attrs
import numpy as np
from numpy.typing import NDArray

from neuralib.util.util_type import PathLike
from neuralib.util.util_verbose import fprint

__all__ = [
    'SIGNAL_TYPE',
    'CALCIUM_TYPE',
    'Suite2pGUIOptions',
    'Suite2pRoiStat',
    'Suite2PResult'
]

SIGNAL_TYPE = Literal["df_f", "spks"]
CALCIUM_TYPE = Literal['baseline', 'transient']


class Suite2pGUIOptions(TypedDict, total=False):
    """ Suite2p GUI setting.

    .. seealso:: `<https://suite2p.readthedocs.io/en/latest/settings.html>`_"""
    look_one_level_down: float
    fast_disk: str
    delete_bin: bool
    mesoscan: bool
    bruker: bool
    h5py: list
    h5py_key: str
    save_path0: str
    save_folder: str
    subfolders: list
    move_bin: bool
    nplanes: int
    nchannels: int
    functional_chan: int
    tau: float
    fs: float
    force_sktiff: bool
    frames_include: int
    multiplane_parallel: float
    preclassify: float
    save_mat: bool
    save_NWB: float
    combined: float
    aspect: float
    do_bidiphase: bool
    bidiphase: float
    bidi_corrected: bool
    do_registration: int
    two_step_registration: float
    keep_movie_raw: bool
    nimg_init: int
    batch_size: int
    maxregshift: float
    align_by_chan: int
    reg_tif: bool
    reg_tif_chan2: bool
    subpixel: int
    smooth_sigma_time: float
    smooth_sigma: float
    th_badframes: float
    norm_frames: bool
    force_refImg: bool
    pad_fft: bool
    nonrigid: bool
    block_size: tuple[int, int]
    snr_thresh: float
    maxregshiftNR: float
    oneP_reg: bool  # 1Preg
    spatial_hp: int
    spatial_hp_reg: float
    spatial_hp_detect: int
    pre_smooth: float
    spatial_taper: float
    roidetect: bool
    spikedetect: bool
    anatomical_only: float
    sparse_mode: bool
    diameter: float
    spatial_scale: float
    connected: bool
    nbinned: int
    max_iterations: int
    threshold_scaling: float
    max_overlap: float
    high_pass: float
    denoise: bool
    soma_crop: bool
    neuropil_extract: bool
    inner_neuropil_radius: float
    min_neuropil_pixels: int
    lam_percentile: float
    allow_overlap: bool
    use_builtin_classifier: bool
    classifier_path: int
    chan2_thres: float
    baseline: str
    win_baseline: float
    sig_baseline: float
    prctile_baseline: float
    neucoeff: int
    suite2p_version: str
    data_path: list[str]
    sbx_ndeadcols: int
    input_format: str
    save_path: str
    ops_path: str
    reg_file: str
    filelist: list[str]
    nframes_per_folder: np.ndarray
    sbx_ndeadrows: int
    meanImg: np.ndarray
    meanImg_chan2: np.ndarray  # if chan_2
    nframes: int
    Ly: int
    Lx: int
    date_proc: datetime.datetime
    refImg: np.ndarray
    rmin: int
    rmax: int
    yblock: list[np.ndarray]
    xblock: list[np.ndarray]
    nblocks: list[int]
    NRsm: np.ndarray
    yoff: np.ndarray
    xoff: np.ndarray
    corrXY: np.ndarray
    yoff1: np.ndarray
    xoff1: np.ndarray
    corrXY1: np.ndarray
    badframes: np.ndarray
    yrange: list[int]
    xrange: list[int]
    tPC: np.ndarray
    regPC: np.ndarray
    regDX: np.ndarray
    Lyc: int
    Lxc: int
    max_proj: np.ndarray
    Vmax: np.ndarray
    ihop: np.ndarray
    Vsplit: np.ndarray
    Vcorr: np.ndarray
    Vmap: list[np.ndarray]
    spatscale_pix: np.ndarray
    meanImgE: np.ndarray
    timing: dict[str, float]


class Suite2pRoiStat(TypedDict, total=False):
    """Suite2p GUI imaging.

    .. seealso:: `<https://suite2p.readthedocs.io/en/latest/outputs.html#stat-npy-fields>`_
    """
    ypix: np.ndarray
    xpix: np.ndarray
    lam: np.ndarray
    med: list[int, int]
    footprint: float
    mrs: float
    mrs0: float
    compact: float
    solidity: float
    npix: int
    npix_soma: int
    soma_crop: np.ndarray
    overlap: np.ndarray
    radius: float
    aspect_ratio: float
    npix_norm_no_crop: float
    npix_norm: float
    skew: float
    std: float


# ============== #
# Suite2P Result #
# ============== #

@final
@attrs.frozen
class Suite2PResult:
    """suite2p result container

    `Dimension parameters`:

        N: number of neurons

        F: number pf frames
    """
    directory: Path
    """directory contain all the s2p output files"""

    F: np.ndarray
    """transient activity 2D array ``(N, F)``"""

    FNeu: np.ndarray
    """neuropil activity 2D array ``(N, F)``"""

    spks: np.ndarray
    """deconvolved activity 2D array ``(N, F)``"""

    stat: NDArray[Suite2pRoiStat]
    """GUI imaging after registration, i.e., x, ypixel ...: np.ndarray with shape: ``(N, )``"""

    ops: Suite2pGUIOptions
    """GUI options"""

    iscell: np.ndarray
    """cell probability for each ROI ``(N, 2)``"""

    redcell: np.ndarray | None
    """red cell probability 2D array ``(N, 2)``"""

    redcell_threshold: float | None
    """red cell probability threshold"""

    runtime_frate_check: float | None = attrs.field(default=None)
    """If not None, check frame rate lower bound"""

    def __attrs_post_init__(self):
        if self.runtime_frate_check is not None:
            self._check_frame_rate()

    def _check_frame_rate(self):
        """User specific check suite2p config is correct"""
        if self.fs * self.n_plane > self.runtime_frate_check:
            fprint(f'the fr: {self.fs} and n_etl: {self.n_plane} might not set properly in suite2p,'
                   f'check output ops.json', vtype='error')
            raise RuntimeError('fs and n_plane are not set properly in suite2p')

    @classmethod
    def launch_gui(cls, directory: PathLike) -> None:
        """
        launch the suite2p GUI

        :param directory: directory contain all the s2p output files. e.g., <SUITE2P_OUTPUT>/suite2p/plane<P>
        :return:
        """
        from suite2p.gui import gui2p

        if not isinstance(directory, Path):
            directory = Path(directory)
        gui2p.run(str(directory / 'stat.npy'))

    @classmethod
    def load(
            cls,
            directory: PathLike,
            cell_prob: bool | float = 0.5,
            red_cell_threshold: float = 0.65,
            channel: int = 0,
            runconfig_frate: float | None = 30.0,
    ) -> Suite2PResult:
        """
        Load suite2p result from directory

        :param directory: directory contain all the s2p output files.
                e.g., <SUITE2P_OUTPUT>/suite2p/plane<P>
        :param cell_prob: cell probability,
                    bool type: use the binary criteria in GUI output
                    float type: value in ``iscell[:, 1]``
        :param red_cell_threshold: red cell threshold
        :param channel: channel (PMT) number for the functional channel.
                    i.e., 0 if GCaMP, 1 if jRGECO in scanbox setting
        :param runconfig_frate: if not None, check frame rate lower-bound to make sure the s2p runconfig
        :return: :class:`Suite2PResult`
        """
        if not isinstance(directory, Path):
            directory = Path(directory)

        if channel == 0:
            F = np.load(directory / 'F.npy', allow_pickle=True)
            FNeu = np.load(directory / 'Fneu.npy', allow_pickle=True)
            spks = np.load(directory / 'spks.npy')
            stat = np.load(directory / 'stat.npy', allow_pickle=True)
            ops = np.load(directory / 'ops.npy', allow_pickle=True).tolist()
            iscell = np.load(directory / 'iscell.npy', allow_pickle=True)
            r = directory / 'redcell.npy'
            if r.exists():
                redcell = np.load(r, allow_pickle=True)
            else:
                redcell = None

        elif channel == 1:  # second channel
            F = np.load(directory / 'F_chan2.npy', allow_pickle=True)
            FNeu = np.load(directory / 'Fneu_chan2.npy', allow_pickle=True)
            spks = np.load(directory / 'spks.npy')
            stat = np.load(directory / 'stat.npy', allow_pickle=True)
            ops = np.load(directory / 'ops.npy', allow_pickle=True).tolist()
            iscell = np.load(directory / 'iscell.npy', allow_pickle=True)
            redcell = None
        else:
            raise IndexError(f'{channel} unknown')

        #
        if cell_prob is True:
            x = iscell[:, 0] == 1
            F = F[x]
            FNeu = FNeu[x]
            spks = spks[x]
            stat = stat[x]
            iscell = iscell[x]
            if redcell is not None:
                redcell = redcell[x]

        elif isinstance(cell_prob, float):
            x = iscell[:, 1] >= cell_prob
            F = F[x]
            FNeu = FNeu[x]
            spks = spks[x]
            stat = stat[x]
            iscell = iscell[x]
            if redcell is not None:
                redcell = redcell[x]
        else:
            raise TypeError(f'invalid type: {type(cell_prob)}')

        return Suite2PResult(
            directory,
            F,
            FNeu,
            spks,
            stat,
            ops,
            iscell,
            redcell,
            red_cell_threshold if channel == 1 else None,
            runconfig_frate
        )

    @property
    def has_chan2(self) -> bool:
        """if has a second channel"""
        if self.ops['nchannels'] == 2:
            return True
        return False

    @property
    def n_neurons(self) -> int:
        """number of neurons after load.
        could be less than GUI ROI number if use higher cell_prob in
        :meth:`~neuralib.calimg.suite2p.core.Suite2PResult.load()`"""
        return self.F.shape[0]

    @property
    def n_frame(self) -> int:
        """number of frame number"""
        return self.F.shape[1]

    @property
    def cell_prob(self) -> np.ndarray:
        """probability that the ROI is a cell based on the default classifier"""
        return self.iscell[:, 1]

    @property
    def n_red_neuron(self) -> int:
        """number of identified neuron based on red cell threshold"""
        if self.has_chan2:
            return np.count_nonzero(self.red_cell_prob >= self.redcell_threshold)
        else:
            raise RuntimeError('no channel 2')

    @property
    def red_cell_prob(self) -> np.ndarray | None:
        """red cell probability"""
        if self.redcell is None:
            return None
        return self.redcell[:, 1]

    @property
    def signal_baseline(self) -> float:
        """Gaussian filter width in seconds"""
        return self.ops['sig_baseline']

    @property
    def window_baseline(self) -> float:
        """window for max/min filter in seconds"""
        return self.ops['win_baseline']

    @property
    def fs(self) -> float:
        """suite2p approximate frame rate per plane, exact value should be checked in .sbx or .mat"""
        return self.ops['fs']

    @property
    def neucoeff(self) -> float:
        """neuropil coefficient, normally should be ~0.7"""
        return self.ops['neucoeff']

    @property
    def prctile_baseline(self) -> float:
        """percentile of trace to use as baseline if ops['baseline'] = constant_percentile"""
        return self.ops['prctile_baseline']

    @property
    def n_plane(self) -> int:
        """number of optical plane"""
        return self.ops['nplanes']

    @property
    def image_width(self) -> int:
        """image width (in pixel)"""
        return self.ops['Lx']

    @property
    def image_height(self) -> int:
        """image height (in pixel)"""
        return self.ops['Ly']

    @property
    def image_mean(self) -> np.ndarray:
        """mean image for chan0(1st)"""
        return self.ops['meanImg'].T

    @property
    def image_mean_ch2(self) -> np.ndarray:
        """mean image for chan1(2nd)"""
        return self.ops['meanImg_chan2'].T

    @property
    def indicator_tau(self) -> float:
        """The timescale of the sensor (in seconds)"""
        return self.ops['tau']

    @property
    def rigid_x_offsets(self) -> np.ndarray:
        """x-shifts of recording at each timepoint (F, )"""
        return self.ops['xoff']

    @property
    def rigid_y_offsets(self) -> np.ndarray:
        """y-shifts of recording at each timepoint (F, )"""
        return self.ops['yoff']

    @property
    def rigid_xy_offset(self) -> np.ndarray:
        """peak of phase correlation between frame and reference image at each timepoint"""
        return self.ops['corrXY']

    @property
    def nonrigid_x_offsets(self) -> np.ndarray:
        """(frames, block_size)"""
        return self.ops['xoff1']

    @property
    def nonrigid_y_offsets(self) -> np.ndarray:
        return self.ops['yoff1']

    @property
    def nonrigid_xy_offsets(self) -> np.ndarray:
        return self.ops['corrXY1']

    @classmethod
    def load_total_neuron_number(cls,
                                 directory: Path,
                                 cell_prob: float | None = 0.5) -> int:
        """
        load number of neuron based on iscell.npy

        :param directory: directory contains the iscell.npy
        :param cell_prob: cell probability,
                    bool type: use the binary criteria in GUI output
                    float type: value in ``iscell[:, 1]``
        :return:
        """
        iscell = np.load(directory / 'iscell.npy', allow_pickle=True)
        if cell_prob is None:
            return np.count_nonzero(iscell[:, 0] == 1)
        else:
            return np.count_nonzero(iscell[:, 1] >= cell_prob)

    def get_rois_pixels(self) -> np.ndarray:
        """ROIs pixel (N, 2)"""
        ret = np.zeros((self.n_neurons, 2))
        for i in range(self.n_neurons):
            x = np.mean(self.stat[i]['xpix'])
            y = np.mean(self.stat[i]['ypix'])
            ret[i] = np.array([x, y])

        return ret
