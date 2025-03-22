import datetime
from pathlib import Path
from typing import Literal, TypedDict, final

import attrs
import numpy as np
import polars as pl
from typing_extensions import Self

from neuralib.imaging.cellular import CellularCoordinates
from neuralib.typing import PathLike
from neuralib.util.deprecation import deprecated_func
from neuralib.util.verbose import fprint

__all__ = [
    'SIGNAL_TYPE',
    'CALCIUM_TYPE',
    #
    'Suite2pGUIOptions',
    'Suite2pRoiStat',
    'Suite2PResult',
    #
    'get_s2p_coords'
]

SIGNAL_TYPE = Literal['df_f', 'spks']
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

        W: image width

        H: image height
    """
    directory: Path
    """Directory contain all the s2p output files"""

    f_raw: np.ndarray
    """Fluorescence traces 2D array. `Array[float, [N, F]]`"""

    f_neu: np.ndarray
    """Neuropil fluorescence traces 2D array. `Array[float, [N, F]]`"""

    spks: np.ndarray
    """Deconvolved activity 2D array. `Array[float, [N, F]]`"""

    stat: np.ndarray
    """GUI imaging after registration, i.e., x, ypixel., etc. `Array[Suite2pRoiStat, N]`"""

    ops: Suite2pGUIOptions
    """GUI options"""

    iscell: np.ndarray
    """Cell probability for each ROI. `Array[float, [N, 2]]`"""

    cell_prob_thres: float | None
    """Cell probability threshold for loading the data"""

    redcell: np.ndarray | None = attrs.field(default=None)
    """Red cell probability 2D array. `Array[float, [N, 2]]`"""

    redcell_threshold: float | None = attrs.field(default=None)
    """Red cell probability threshold"""

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
    @deprecated_func(removal_version='0.5.0', remarks='lightening dependency: suite2p, use an separated env')
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
    def load(cls, directory: PathLike,
             cell_prob_thres: float | None = 0.5,
             red_cell_threshold: float = 0.65,
             channel: int = 0,
             runconfig_frate: float | None = 30.0) -> Self:
        """
        Load suite2p result from directory

        :param directory: Directory contain all the s2p output files. e.g., \*/suite2p/plane[P]
        :param cell_prob_thres: Cell probability. If float type, mask for the value in ``iscell[:, 1]``.
                    If None, use the binary criteria in GUI output
        :param red_cell_threshold: Red cell threshold
        :param channel: channel (PMT) Number for the functional channel. i.e., 0 if GCaMP, 1 if jRGECO in scanbox setting
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
        if cell_prob_thres is None:
            x = iscell[:, 0] == 1
        elif isinstance(cell_prob_thres, float):
            x = iscell[:, 1] >= cell_prob_thres
        else:
            raise TypeError(f'invalid type: {type(cell_prob_thres)}')

        F = F[x]
        FNeu = FNeu[x]
        spks = spks[x]
        stat = stat[x]
        iscell = iscell[x]
        if redcell is not None:
            redcell = redcell[x]

        return Suite2PResult(
            directory,
            F,
            FNeu,
            spks,
            stat,
            ops,
            iscell,
            cell_prob_thres,
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
        :meth:`~neuralib.imaging.suite2p.core.Suite2PResult.load()`"""
        return self.f_raw.shape[0]

    @property
    def n_frame(self) -> int:
        """number of frame number"""
        return self.f_raw.shape[1]

    @property
    def cell_prob(self) -> np.ndarray:
        """probability that the ROI is a cell based on the default classifier. `Array[float, N]`"""
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
        """red cell probability, `Array[float, N]`"""
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
        """mean image for chan0(1st). `Array[float, [H, W]]`"""
        return self.ops['meanImg'].T

    @property
    def image_mean_ch2(self) -> np.ndarray:
        """mean image for chan1(2nd). `Array[float, [H, W]]`"""
        return self.ops['meanImg_chan2'].T

    @property
    def indicator_tau(self) -> float:
        """The timescale of the sensor (in seconds)"""
        return self.ops['tau']

    @property
    def rigid_x_offsets(self) -> np.ndarray:
        """x-shifts of recording at each timepoint. `Array[int, F]`"""
        return self.ops['xoff']

    @property
    def rigid_y_offsets(self) -> np.ndarray:
        """y-shifts of recording at each timepoint. `Array[int, F]`"""
        return self.ops['yoff']

    @property
    def rigid_xy_offset(self) -> np.ndarray:
        """peak of phase correlation between frame and reference image at each timepoint. `Array[float, F]`"""
        return self.ops['corrXY']

    @property
    def nonrigid_x_offsets(self) -> np.ndarray:
        """(frames, block_size). `Array[float, F]`"""
        return self.ops['xoff1']

    @property
    def nonrigid_y_offsets(self) -> np.ndarray:
        """`Array[float, F]`"""
        return self.ops['yoff1']

    @property
    def nonrigid_xy_offsets(self) -> np.ndarray:
        """`Array[float, F]`"""
        return self.ops['corrXY1']

    @classmethod
    def load_total_neuron_number(cls,
                                 directory: Path,
                                 cell_prob: float | None = 0.5) -> int:
        """
        Load number of neuron based on iscell.npy

        :param directory: directory contains the iscell.npy
        :param cell_prob: cell probability,
                    bool type: use the binary criteria in GUI output
                    float type: value in ``iscell[:, 1]``
        :return: Number of neurons
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

    def get_neuron_id_mapping(self) -> pl.DataFrame:
        """
        Retrieves a mapping between neuron IDs and their corresponding raw indices
        based on whether the cell detection probabilities meet a specified threshold.
        If no cell detection probabilities are provided, the mapping assumes all
        indices are valid neurons.

        :return: A Polars DataFrame containing two columns: `neuron_id` and  `raw_index`.
        """
        n = np.arange(len(self.f_raw))
        if self.cell_prob is None:
            return pl.DataFrame([n, n], schema=['neuron_id', 'raw_index'], orient='col')
        else:
            iscell = np.load(self.directory / 'iscell.npy', allow_pickle=True)
            mx = np.nonzero(iscell[:, 1] >= self.cell_prob)[0]
            return pl.DataFrame([n, mx], schema=['neuron_id', 'raw_index'], orient='col')


def get_s2p_coords(s2p: Suite2PResult,
                   neuron_list: int | list[int] | slice | np.ndarray | None,
                   plane_index: int,
                   factor: float) -> CellularCoordinates:
    """
    Get the suite2p coordinates of all cells.

    :param s2p: ``Suite2PResult``
    :param neuron_list: neuron index or index list/arr. If None, then load all neurons
    :param plane_index: optic plane index
    :param factor: pixel to mm factor
    :return: :class:`~neuralib.imaging.cellular_cords.CellularCoordinates`
    """
    if neuron_list is None:
        neuron_list = np.arange(s2p.n_neurons)

    n_neurons = len(neuron_list)
    xpix = np.zeros(n_neurons)
    ypix = np.zeros(n_neurons)

    for i, n in enumerate(neuron_list):
        xpix[i] = np.mean(s2p.stat[i]['xpix'])
        ypix[i] = np.mean(s2p.stat[i]['ypix'])

    xcord = xpix * factor / 1000  # ap
    ycord = ypix * factor / 1000  # ml

    if isinstance(neuron_list, int):
        src_plane_index = plane_index
    elif isinstance(neuron_list, (list, np.ndarray, slice)):
        src_plane_index = np.full_like(neuron_list, plane_index)
    else:
        raise TypeError('')

    return CellularCoordinates(
        np.array(neuron_list),
        xcord,
        ycord,
        plane_index=plane_index,
        unit='mm',
        source_plane_index=src_plane_index
    )
