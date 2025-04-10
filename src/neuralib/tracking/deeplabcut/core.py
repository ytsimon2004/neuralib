from __future__ import annotations

import pickle
from pathlib import Path
from typing import TypedDict

import polars as pl

from neuralib.typing import PathLike
from neuralib.util.dataframe import DataFrameWrapper

__all__ = [
    'Joint',
    'read_dlc',
    'DeepLabCutDataFrame',
    'JointDataFrame',
    'DeepLabCutMeta',
    'DeepLabCutModelConfig'
]

Joint = str
"""Joint name"""


def read_dlc(file: PathLike, meta_file: PathLike | None = None) -> DeepLabCutDataFrame:
    """
    load DeepLabCut result from file

    :param file: DeepLabCut result filepath. supports both ``.h5`` and ``.csv``
    :param meta_file: Optional DeepLabCut meta filepath. should be the ``.pickle``
    :return:
    """
    file = Path(file)
    meta_file = Path(meta_file)
    meta = _load_meta(meta_file)

    match file.suffix:
        case '.h5' | '.hdf5':
            df = _load_dlc_h5_table(file)
        case '.csv':
            df = _load_dlc_csv(file)
        case _:
            raise ValueError(f'file: {file} is not supported')

    return DeepLabCutDataFrame(df, meta=meta, filtered=('filtered' in file.name))


class DeepLabCutDataFrame(DataFrameWrapper):
    """
    DeepLabCut DataFrame ::

        ┌───────────┬───────────┬───────────┬───────────┬───┬───────────┬───────────┬───────────┬──────────┐
        │ Nose_x    ┆ Nose_y    ┆ Nose_like ┆ EarL_x    ┆ … ┆ TailMid_l ┆ TailEnd_x ┆ TailEnd_y ┆ TailEnd_ │
        │ ---       ┆ ---       ┆ lihood    ┆ ---       ┆   ┆ ikelihood ┆ ---       ┆ ---       ┆ likeliho │
        │ f64       ┆ f64       ┆ ---       ┆ f64       ┆   ┆ ---       ┆ f64       ┆ f64       ┆ od       │
        │           ┆           ┆ f64       ┆           ┆   ┆ f64       ┆           ┆           ┆ ---      │
        │           ┆           ┆           ┆           ┆   ┆           ┆           ┆           ┆ f64      │
        ╞═══════════╪═══════════╪═══════════╪═══════════╪═══╪═══════════╪═══════════╪═══════════╪══════════╡
        │ 57.907318 ┆ 512.54742 ┆ 0.999679  ┆ 77.701355 ┆ … ┆ 0.999904  ┆ 257.71426 ┆ 561.89660 ┆ 0.999961 │
        │           ┆ 4         ┆           ┆           ┆   ┆           ┆ 4         ┆ 6         ┆          │
        │ 57.907318 ┆ 516.79528 ┆ 0.999688  ┆ 77.701355 ┆ … ┆ 0.999923  ┆ 257.71426 ┆ 562.05725 ┆ 0.999954 │
        │ …         ┆ …         ┆ …         ┆ …         ┆ … ┆ …         ┆ …         ┆ …         ┆ …        │
        │ 94.259621 ┆ 43.849434 ┆ 0.973851  ┆ 106.33532 ┆ … ┆ 0.998977  ┆ 87.477776 ┆ 257.11996 ┆ 0.999937 │
        │ 94.294357 ┆ 44.340511 ┆ 0.965436  ┆ 106.45220 ┆ … ┆ 0.999604  ┆ 87.223534 ┆ 258.46600 ┆ 0.999912 │
        └───────────┴───────────┴───────────┴───────────┴───┴───────────┴───────────┴───────────┴──────────┘

    """

    def __init__(self, df: pl.DataFrame,
                 meta: DeepLabCutMeta | None, *,
                 filtered: bool):
        """
        :param df: DeepLabCut result dataframe
        :param meta: :attr:`~DeepLabCutMeta`
        :param filtered: whether the results has already been filtered
        """
        self._df = df
        self._meta = meta
        self._filtered = filtered

    def __repr__(self):
        raise repr(self.dataframe())

    def dataframe(self, dataframe: pl.DataFrame = None, may_inplace=True):
        if dataframe is None:
            return self._df
        else:
            return DeepLabCutDataFrame(dataframe, meta=self._meta, filtered=self._filtered)

    @property
    def default_filtered(self) -> bool:
        """whether default filtered when running the deeplabcut"""
        return self._filtered

    @property
    def meta(self) -> DeepLabCutMeta:
        """:attr:`~neuralib.tracking.deeplabcut.DeepLabCutMeta`"""
        return self._meta

    @property
    def model_config(self) -> DeepLabCutModelConfig:
        """:attr:`~neuralib.tracking.deeplabcut.DeepLabCutModelConfig`"""
        return self.meta['model_config']

    @property
    def fps(self) -> float:
        """frame per second, meta data required"""
        return self.meta['fps']

    @property
    def nframes(self) -> int:
        """number of frames"""
        return self.meta['nframes']

    @property
    def joints(self) -> list[Joint]:
        """list of labelled joints"""
        return self.meta['model_config']['all_joints_names']

    def get_joint(self, joint: Joint) -> JointDataFrame:
        """get specific joint"""
        cols = ('x', 'y', 'likelihood')
        df = self.select([f'{joint}_{col}' for col in cols]).dataframe()
        return JointDataFrame(df)


class JointDataFrame(DataFrameWrapper):
    """
    Dataframe from a specific joint ::

        ┌───────────┬────────────┬─────────────────┐
        │ Nose_x    ┆ Nose_y     ┆ Nose_likelihood │
        │ ---       ┆ ---        ┆ ---             │
        │ f64       ┆ f64        ┆ f64             │
        ╞═══════════╪════════════╪═════════════════╡
        │ 57.907318 ┆ 512.547424 ┆ 0.999679        │
        │ 57.907318 ┆ 516.795288 ┆ 0.999688        │
        │ 57.907318 ┆ 519.56311  ┆ 0.999449        │
        │ 56.733799 ┆ 522.204224 ┆ 0.999161        │
        │ 53.546089 ┆ 525.24939  ┆ 0.999518        │
        │ …         ┆ …          ┆ …               │
        │ 94.259621 ┆ 43.849434  ┆ 0.973851        │
        │ 94.294357 ┆ 44.111595  ┆ 0.980125        │
        │ 94.8013   ┆ 44.340511  ┆ 0.963981        │
        │ 94.294357 ┆ 44.340511  ┆ 0.947905        │
        │ 94.294357 ┆ 44.340511  ┆ 0.965436        │
        └───────────┴────────────┴─────────────────┘
    """

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def __repr__(self):
        return repr(self.dataframe())

    def dataframe(self, dataframe: pl.DataFrame = None, may_inplace=True):
        if dataframe is None:
            return self._df
        else:
            return JointDataFrame(dataframe)


class DeepLabCutMeta(TypedDict):
    """DeepLabCut model metadata"""
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
    """DeepLabCut model configuration"""
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
