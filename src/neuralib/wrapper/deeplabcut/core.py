from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import NamedTuple, Optional, Literal, Union, TypedDict

import numpy as np
import polars as pl
from scipy.ndimage import gaussian_filter1d

from neuralib.plot import plot_figure
from neuralib.util.util_type import PathLike
from neuralib.wrapper.deeplabcut.util import remove_jumps, interpolate_gaps, compute_velocity

__all__ = [
    'VideoConfig',
    'DlcResult',
    'DlcTrackPart'
]


# ============ #
# Video Config #
# ============ #

class VideoConfig(TypedDict):
    xy_pixel: tuple[int, int]
    width: int
    height: int
    length_unit: Literal['cm', 'mm']
    fps: float
    """in hz"""


def get_default_config() -> VideoConfig:
    return VideoConfig(
        xy_pixel=(1024, 1024),
        width=300,
        height=300,
        length_unit='mm',
        fps=30
    )


def read_default_config() -> VideoConfig:
    """TODO from video"""
    pass


def get_pixel2length_factor(config: VideoConfig) -> float:
    """"rough calculation"""
    wp = config['width'] / config['xy_pixel'][0]
    hp = config['height'] / config['xy_pixel'][1]
    return (wp + hp) / 2


# ========== #
# DLC Result #
# ========== #

@dataclasses.dataclass
class DlcResult:
    prefix: str
    model_name: str
    name: str
    code: str
    with_filter: bool

    config: VideoConfig

    data: pl.DataFrame
    """
    ┌────────────┬────────────┬─────────────────┬────────────┬────────────┬────────────────┬───────┐
    │ blue_x     ┆ blue_y     ┆ blue_likelihood ┆ red_x      ┆ red_y      ┆ red_likelihood ┆ *valid│
    │ ---        ┆ ---        ┆ ---             ┆ ---        ┆ ---        ┆ ---            ┆ ---   │
    │ f64        ┆ f64        ┆ f64             ┆ f64        ┆ f64        ┆ f64            ┆ bool  │
    ╞════════════╪════════════╪═════════════════╪════════════╪════════════╪════════════════╪═══════╡
    │ *NaN       ┆ NaN        ┆ 0.8             ┆ NaN        ┆ NaN        ┆ 1.0            ┆ false │
    │ 422.874268 ┆ 266.782867 ┆ 1.0             ┆ 422.117218 ┆ 272.525543 ┆ 1.0            ┆ true  │
    │ …          ┆ …          ┆ …               ┆ …          ┆ …          ┆ …              ┆ …     │
    │ 350.320679 ┆ 296.086517 ┆ 1.0             ┆ 357.244385 ┆ 291.834351 ┆ 1.0            ┆ true  │
    └────────────┴────────────┴─────────────────┴────────────┴────────────┴────────────────┴───────┘
    """

    parts: tuple[str, ...]

    valid_mask: Optional[np.ndarray] = dataclasses.field(init=False, default=None)
    invalid_frame_index: set[int] = dataclasses.field(default_factory=set)
    _col_order: tuple[str] = dataclasses.field(init=False, default=('x', 'y', 'likelihood'))

    def __post_init__(self):
        mask = np.ones(self.data.shape[0])
        invalid = np.array(list(self.invalid_frame_index)).astype(int)
        mask[invalid] = 0
        self.valid_mask = np.array(mask, dtype=bool)

    @classmethod
    def load(cls, filepath: PathLike,
             config: Optional[VideoConfig] = None) -> 'DlcResult':
        file = Path(filepath)
        p, m, n, c = file.stem.split('_')
        with_filter = True if c.endswith('filter') else False

        with file.open() as f:
            f.readline()
            parts = f.readline().strip().split(',')[1::3]

        new_col = ['']
        for p in parts:
            for it in cls._col_order:
                new_col.append(f'{p}_{it}')
        #
        data = pl.read_csv(filepath, skip_rows=3, has_header=False, new_columns=new_col)[:, 1:]

        #
        if config is None:
            config = get_default_config()

        #
        factor = get_pixel2length_factor(config)
        for c in ['x', 'y']:
            for p in parts:
                data = data.with_columns(pl.col(f'{p}_{c}') * factor)

        return DlcResult(p, m, n, c, with_filter, config, data, tuple(parts))

    def with_global_likelihood_filtered(self,
                                        threshold: float = 0.9,
                                        interp_nan: bool = True,
                                        interp_gap: float = 0.5) -> 'DlcResult':
        """For all parts
        replace xy as nan if any parts with low likelihood
        ** Note if even interp_nan, the non-interpolate nan need to be further processed in the caller site.
        """
        data = self.data.with_columns(
            pl.concat_list([pl.col(f'{p}_likelihood') >= threshold for p in self.parts]).list.all().alias('valid')
        )

        invalid = data.select(~pl.col('valid')).to_numpy().ravel()
        self.invalid_frame_index.update(np.nonzero(invalid)[0])

        data = data.with_columns(
            *([
                  pl.when(pl.col('valid')).then(pl.col(f'{p}_x')).otherwise(np.nan)
                  for p in self.parts
              ] + [
                  pl.when(pl.col('valid')).then(pl.col(f'{p}_y')).otherwise(np.nan)
                  for p in self.parts
              ])
        )

        if interp_nan:
            for p in self.parts:
                xy = data.select([pl.col(f'{p}_x', f'{p}_y')]).to_numpy()
                _xy = interpolate_gaps(self.time, xy, interp_gap)
                for i, c in enumerate(['x', 'y']):
                    data = data.with_columns(pl.Series(_xy[:, i]).alias(f'{p}_{c}'))

        return dataclasses.replace(self, data=data)

    def average_across(self, parts: list[str]) -> DlcTrackPart:
        if any([p not in self.parts for p in parts]):
            raise ValueError('')

        data = self.data.select(
            avg_x=pl.concat_list([f'{p}_x' for p in parts]).list.mean(),
            avg_y=pl.concat_list([f'{p}_y' for p in parts]).list.mean(),
            likelihood=pl.concat_list([f'{p}_likelihood' for p in parts]).list.mean()
        ).to_numpy()

        return DlcTrackPart(
            self,
            parts,
            data[:, :2],
            data[:, 2],
            self.config['fps']
        )

    def __getitem__(self, part: str) -> DlcTrackPart:
        if part not in self.parts:
            raise KeyError(f'unknown {part}, should be in {self.parts}')

        ret = self.data.select([f'{part}_{i}' for i in self._col_order]).to_numpy()

        return DlcTrackPart(
            self,
            part,
            ret[:, :2],
            ret[:, 2],
            self.fps
        )

    @property
    def n_frames(self) -> int:
        return self.data.shape[0]

    @property
    def time(self) -> np.ndarray:
        """Assume stable frame rate"""
        return np.linspace(0, self.total_time, self.n_frames)

    @property
    def total_time(self) -> float:
        return self.n_frames * self.fps

    @property
    def fps(self) -> float:
        return self.config['fps']

    @fps.setter
    def fps(self, value: float):
        if value != self.fps:
            print(f'reset video fps {self.fps} -> {value}')
        self.config['fps'] = value


# ========== #
# Track Part #
# ========== #

class DlcTrackPart(NamedTuple):
    source: 'DlcResult'

    parts: Union[str, list[str]]  # 1|P
    xy: np.ndarray  # (N, 2)
    likelihood: np.ndarray
    fps: float

    #
    processed: bool = False
    likelihood_removal_index: list[int] = []
    jump_removal_index: list[list[int]] = []

    @property
    def n_frames(self) -> int:
        return self.xy.shape[0]

    @property
    def time(self) -> np.ndarray:
        return np.linspace(0, self.total_time, self.n_frames)

    @property
    def total_time(self) -> float:
        return self.n_frames * self.fps

    @property
    def speed(self) -> np.ndarray:
        return np.abs(compute_velocity(self.xy, self.fps))

    @property
    def direction(self) -> np.ndarray:
        return np.angle(compute_velocity(self.xy, self.fps), deg=True)

    def with_likelihood_filtered(self, threshold: float = 0.95) -> 'DlcTrackPart':
        xy = self.xy.copy()
        invalid = np.nonzero(self.likelihood < threshold)[0]
        xy[invalid] = np.nan
        return self._replace(xy=xy, likelihood_removal_index=invalid.tolist())

    def with_jump_removal(self, jump_size: int = 20, duration: float = 0.1) -> 'DlcTrackPart':
        ret, jump = remove_jumps(self.time, self.xy, jump_size, duration)
        return self._replace(xy=ret, jump_removal_index=jump)

    def interpolate_gaps(self, interpolate_gap: float) -> 'DlcTrackPart':
        xy = interpolate_gaps(self.time, self.xy, interpolate_gap)
        return self._replace(xy=xy)

    def with_proc(self, jump_size: int = 20,
                  duration: float = 0.1,
                  interpolate_gap: float = 10) -> 'DlcTrackPart':
        if self.processed:
            return self
        return (self.with_likelihood_filtered()
                .with_jump_removal(jump_size, duration)
                .interpolate_gaps(interpolate_gap)
                ._replace(processed=True))

    def _kernel_filter(self,
                       xy: np.ndarray,
                       gaussian_kernel: Optional[float] = None,
                       run_kalman_filter: bool = False,
                       **kalman_kw) -> np.ndarray:

        if kalman_kw is None:
            kalman_kw = dict(P=200, R=0.5, Q=2000)  # TODO opt and not test yet

        if gaussian_kernel is not None:
            xy = gaussian_filter1d(xy, gaussian_kernel, axis=0)

        if run_kalman_filter:
            from rscvp.util.util_signal import run_kalman_2d
            speed = np.vstack([self.speed, self.speed]).T
            xs = run_kalman_2d(xy, speed, dt=1 / self.fps, **kalman_kw)[0]
            xy = xs[:, :2]

        return xy

    def plot_trajectory_occ(self, **kwargs):
        import seaborn as sns
        if not self.processed:
            xy = self.with_proc().xy
        else:
            xy = self.xy

        xy = self._kernel_filter(xy, **kwargs)
        with plot_figure(None, 1, 2) as ax:
            ax[0].plot(xy[:, 0], xy[:, 1])
            sns.histplot(x=xy[:, 0], y=xy[:, 1], cmap='twilight_shifted',
                         bins=100, ax=ax[1], cbar=True, cbar_kws=dict(shrink=0.25))
