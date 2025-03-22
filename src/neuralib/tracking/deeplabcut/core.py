from __future__ import annotations

import pickle
from pathlib import Path
from typing import TypedDict, NamedTuple

import h5py
import numpy as np
import polars as pl
from neuralib.typing import PathLike
from typing_extensions import Self

__all__ = [
    'Joint',
    'DeepLabCutMeta',
    'DeepLabCutModelConfig',
    'DeepLabCutResult',
    'load_dlc_result'
]

Joint = str


class DeepLabCutMeta(TypedDict):
    start: float
    stop: float
    run_duration: float
    Scorer: str
    model_config: DeepLabCutModelConfig
    fps: float
    batch_size: int
    frame_dimensions: tuple[int, int]
    nframes: int
    iteration: int
    training_set_fraction: float
    cropping: bool
    cropping_parameters: list[tuple[float, float, float, float]]


class DeepLabCutModelConfig(TypedDict):
    stride: float
    weigh_part_predictions: bool
    weigh_negatives: bool
    fg_fraction: float
    mean_pixel: list[float]
    shuffle: bool
    snapshot_prefix: str
    log_dir: str
    global_scale: float
    location_refinement: bool
    locref_stdev: float
    locref_loss_weight: float
    locref_huber_loss: bool
    optimizer: str
    intermediate_supervision: bool
    intermediate_supervision_layer: int
    regularize: bool
    weight_decay: float
    crop_pad: int
    scoremap_dir: str
    batch_size: int
    dataset_type: str
    deterministic: bool
    mirror: bool
    pairwise_huber_loss: bool
    weigh_only_present_joints: bool
    partaffinityfield_predict: bool
    pairwise_predict: bool
    all_joints: list[list[int]]
    all_joints_names: list[Joint]
    dataset: str
    init_weights: str
    net_type: str
    num_joints: int
    num_outputs: int


class DeepLabCutJoint(NamedTuple):
    source: DeepLabCutResult
    name: Joint
    dat: pl.DataFrame

    def with_lh_filter(self, lh: float) -> Self:
        expr = pl.col('likelihood') >= lh
        dat = self.dat.with_columns(
            pl.when(expr).then(pl.col('x')).otherwise(np.nan),
            pl.when(expr).then(pl.col('y')).otherwise(np.nan)
        )
        return self._replace(dat=dat)

    @property
    def x(self) -> np.ndarray:
        return self.dat['x'].to_numpy()

    @property
    def y(self) -> np.ndarray:
        return self.dat['y'].to_numpy()

    @property
    def xy(self) -> np.ndarray:
        return self.dat['x', 'y'].to_numpy()

    @property
    def t(self) -> np.ndarray:
        return self.source.time

    @property
    def lh(self) -> np.ndarray:
        return self.dat['likelihood'].to_numpy()


class DeepLabCutResult:

    def __init__(self,
                 dat: pl.DataFrame,
                 meta: DeepLabCutMeta,
                 filtered: bool,
                 time: np.ndarray | None = None):
        """

        :param dat: Deeplabcut results as polars dataframe
        :param meta: Deeplabcut meta typeddict
        :param filtered: If the Deeplabcut results is filtered or not
        :param time: 1D time array for each tracked frames. If None, then assume stable DAQ and calculated from meta.
        """
        self.dat = dat
        self._meta = meta
        self._filtered = filtered

        self._time = time if time is not None else self._default_time()

    def __getitem__(self, item: Joint) -> DeepLabCutJoint:
        """Get data from a specific joint"""
        cols = ('x', 'y', 'likelihood')
        dat = self.dat[[f'{item}_{col}' for col in cols]]
        dat.columns = cols
        return DeepLabCutJoint(self, item, dat)

    @property
    def is_filtered(self) -> bool:
        return self._filtered

    @property
    def meta(self) -> DeepLabCutMeta:
        return self._meta

    @property
    def joints(self) -> list[Joint]:
        """list of labelled joints"""
        return self.meta['model_config']['all_joints_names']

    @property
    def fps(self) -> float:
        return self.meta['fps']

    @property
    def nframes(self) -> int:
        return self.meta['nframes']

    @property
    def time(self) -> np.ndarray:
        if len(self._time) != self.dat.shape[0]:
            raise ValueError('time array has wrong shape, mismatch with dat')
        return self._time

    def _default_time(self):
        """Based on meta file, assume stable frame acquisition"""
        return np.linspace(0, self.nframes / self.fps, self.nframes)

    def with_global_lh_filter(self, lh: float) -> Self:
        """
        With global likelihood filter
        :param lh: likelihood threshold
        :return: ``DeepLabCutResult``
        """
        for j in self.joints:
            self.dat = (
                self.dat
                .with_columns((pl.col(f'{j}_likelihood') >= lh).alias('valid'))
                .with_columns(pl.when(pl.col('valid')).then(pl.col(f'{j}_x')).otherwise(np.nan))
                .with_columns(pl.when(pl.col('valid')).then(pl.col(f'{j}_y')).otherwise(np.nan))
            )

        return self


def load_dlc_result(file: PathLike,
                    meta_file: PathLike,
                    time: np.ndarray | None = None) -> DeepLabCutResult:
    """
    Load DeepLabCut result from file

    :param file: DeepLabCut result filepath. supports both `.h5` and `.csv`
    :param meta_file: DeepLabCut meta filepath. should be the `.pickle`. TODO Cannot it be inferred according to file?
    :param time: time array for each sample point. If None, then assume stable DAQ for using total frames and fps info in meta
    :return: ``DeepLabCutResult``
    """
    file = Path(file)
    meta_file = Path(meta_file)
    meta = _load_meta(meta_file)

    if file.suffix in ('.h5', '.hdf5'):
        try:
            import pytables  # noqa: F401
            import pandas  # noqa: F401
            df = _load_dlc_h5_table(file)
        except ImportError:
            df = _load_dlc_h5(file, meta)
    elif file.suffix == '.csv':
        df = _load_dlc_csv(file)
    else:
        raise ValueError(f'Unsupported file type: {file.suffix}')

    return DeepLabCutResult(df, meta, filtered=('filtered' in file.name), time=time)


def _load_dlc_h5_table(file) -> pl.DataFrame:
    import pandas as pd

    df = pd.read_hdf(file)

    scorers = list(df.columns.levels[0])
    bodyparts = list(df.columns.levels[1])
    coords = list(df.columns.levels[2])

    assert len(scorers) == 1
    scorer = scorers[0]

    data = {
        f'{b}_{c}': df[(scorer, b, c)]
        for b in bodyparts
        for c in coords
    }

    ret = pl.DataFrame(data)

    return ret


def _load_dlc_h5(file, meta) -> pl.DataFrame:
    dat = h5py.File(file)['df_with_missing']['table']['values_block_0']
    return pl.from_numpy(dat, schema=[
        f'{joint}_{it}'
        for joint in meta['model_config']['all_joints_names']
        for it in ('x', 'y', 'likelihood')
    ])


def _load_dlc_csv(file) -> pl.DataFrame:
    cols = ['']
    with file.open() as f:
        f.readline()  # skip first line
        parts = f.readline().strip().split(',')[1::3]
        cols.extend([f'{p}_{it}' for p in parts for it in ('x', 'y', 'likelihood')])

    df = pl.read_csv(file, skip_rows=3, has_header=False, new_columns=cols)[:, 1:]

    return df


def _load_meta(meta_file) -> DeepLabCutMeta:
    if meta_file.suffix not in ('.pkl', '.pickle'):
        raise ValueError(f'{meta_file} is not a pickle file')

    # meta
    with meta_file.open('rb') as f:
        meta = pickle.load(f)['data']

    # copy to typeddict
    meta['model_config'] = meta['DLC-model-config file']
    meta['iteration'] = meta['iteration (active-learning)']
    meta['training_set_fraction'] = meta['training set fraction']

    return meta
