import dataclasses
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from scipy.interpolate import interp1d

from neuralib.argp import as_argument, argument
from neuralib.atlas.brainrender.core import BrainReconstructor
from neuralib.atlas.brainrender.util import roi_points_converter
from neuralib.atlas.util import PLANE_TYPE
from neuralib.util.segement import grouped_iter


class ProbeReconstructor(BrainReconstructor):
    DESCRIPTION = 'For probe(s) track reconstruction'

    implant_depth: int = argument('-D', '--depth', required=True, help='implant depth in um')

    dye_label_only: bool = argument('--dye', help='only show the histology dye parts')

    csv_file: Path = as_argument(BrainReconstructor.csv_file).with_options(
        required=True,
        help='csv file after registration using 2dccf pipeline, point numbers equal to probe(s) * 2'
    )
    """Example
    ┌───────────────────────────────────┬─────────┬─────────────┬─────────────┬─────────────┬─────────┐
    │ name                              ┆ acronym ┆ AP_location ┆ DV_location ┆ ML_location ┆ avIndex │
    │ ---                               ┆ ---     ┆ ---         ┆ ---         ┆ ---         ┆ ---     │
    │ str                               ┆ str     ┆ f64         ┆ f64         ┆ f64         ┆ i64     │
    ╞═══════════════════════════════════╪═════════╪═════════════╪═════════════╪═════════════╪═════════╡
    │ Primary visual area layer 6a      ┆ VISp6a  ┆ -3.81       ┆ 1.92        ┆ -3.12       ┆ 191     │
    │ optic radiation                   ┆ or      ┆ -4.08       ┆ 2.33        ┆ -3.12       ┆ 1217    │
    │ Posterolateral visual area layer… ┆ VISpl6a ┆ -4.28       ┆ 2.29        ┆ -3.12       ┆ 198     │
    │ Posterolateral visual area layer… ┆ VISpl5  ┆ -4.52       ┆ 2.17        ┆ -3.12       ┆ 197     │
    │ Subiculum                         ┆ SUB     ┆ -3.93       ┆ 4.36        ┆ -3.3        ┆ 536     │
    │ Entorhinal area medial part dors… ┆ ENTm5   ┆ -4.19       ┆ 4.39        ┆ -3.3        ┆ 515     │
    │ Entorhinal area medial part dors… ┆ ENTm2   ┆ -4.44       ┆ 4.39        ┆ -3.3        ┆ 510     │
    │ Entorhinal area medial part dors… ┆ ENTm1   ┆ -4.66       ┆ 4.29        ┆ -3.3        ┆ 509     │
    └───────────────────────────────────┴─────────┴─────────────┴─────────────┴─────────────┴─────────┘
    """

    cut_plane: PLANE_TYPE = argument(
        '--plane-type', '-P',
        default='coronal',
        help='cutting orientation'
    )

    raw: pl.DataFrame
    data: pl.DataFrame  # sorted

    def post_parsing(self):
        super().post_parsing()
        self.raw = pl.read_csv(self.csv_file)
        self.data = self._sort_dataframe_order(self.raw)

    def load(self):
        self.dye_label_only = True
        probe_dye = self.get_probe_object().shanks
        probe_dye = np.vstack(probe_dye)

        self.dye_label_only = False
        probe_theo = self.get_probe_object().with_theoretical().shanks
        probe_theo = np.vstack(probe_theo)

        self.add_points([probe_dye, probe_theo])

    @cached_property
    def number_shanks(self) -> int:
        return int(self.raw.shape[0] / 2)

    @dataclasses.dataclass
    class ShanksTrack:

        clazz: 'ProbeReconstructor'
        shanks: list[np.ndarray]  # (S, N)

        def __post_init__(self):
            assert len(self.shanks) == self.clazz.number_shanks

        def __len__(self) -> int:
            return len(self.shanks)

        def __getitem__(self, idx: int) -> np.ndarray:
            return self.shanks[idx]

        def with_theoretical(self, interval: int = 250) -> 'ProbeReconstructor.ShanksTrack':
            """theoretical track based on implantation depth / angle"""
            if not self.clazz.dye_label_only:
                crop = self.clazz.crop_outside_brain
                depth = self.clazz.implant_depth
                ret = []
                s1 = self[0]

                for i in range(len(self)):
                    s_cur = self[i]
                    p = crop(s_cur, depth, dv_value=int(s1[0, 1]))

                    ret.append(p)
                    depth += _calc_shank_length_diff(s1, interval)
                    interval += interval
                return dataclasses.replace(self, shanks=ret)

            raise RuntimeError('')

    # TODO test ruler obj?
    def get_probe_object(self) -> ShanksTrack:
        p = roi_points_converter(self.data)

        ext_depth = None if self.dye_label_only else (0, 5000)

        fn = lambda p1, p2: self._shank_extend(np.array([p1, p2]), ext_depth=ext_depth)

        ret = []
        for (surface_idx, tip_idx) in grouped_iter(np.arange(p.shape[0]), 2):
            ret.append(fn(p[surface_idx], p[tip_idx]))

        return self.ShanksTrack(self, ret)

    def _sort_dataframe_order(self, df: pl.DataFrame):
        """probe in correct order"""
        n_shanks = self.number_shanks

        df = (df.sort('DV_location')
              .with_columns(pl.Series(['dorsal_label'] * n_shanks + ['ventral_label'] * n_shanks).alias('probe')))

        if self.cut_plane == 'sagittal':
            probe_order = 'AP_location'
        elif self.cut_plane == 'coronal':
            probe_order = 'ML_location'
        else:
            raise RuntimeError('')

        df = (df.sort(by=['probe', probe_order], descending=[False, True])
              .with_columns(pl.Series(list(range(1, 1 + n_shanks)) * 2).alias('probe_idx'))
              .sort('probe_idx'))

        print(df)

        return df

    def isin_brain(self, shank: np.ndarray) -> np.ndarray:
        """
        determine if the probe points are in the brain

        :param shank: (N', 3)
        :return:
        """
        brain = self.get_atlas_brain_globe()

        ret = []
        for sh in shank:
            try:
                s = brain.structure_from_coords(sh, microns=True)
            except IndexError:
                s = 0
            ret.append(s != 0)

        return np.array(ret, dtype=bool)

    def crop_outside_brain(self, shank: np.ndarray,
                           distance: float,
                           dv_value: Optional[int] = None) -> np.ndarray:
        """
        crop the probe after doing the extension

        :param shank: (N', 3)
        :param distance: depth of insertion (might with an angle, in um). mostly records during the implantation
                    use the depth value that used while implantation to cutoff the bottom line.
        :param dv_value: if None, plot the probe if its in the brain
                if int type, plot the probe if dv larger than this value
        :return:
        """
        shank = shank[shank[:, 1] >= 0]

        if dv_value is None:
            m = self.isin_brain(shank)
        elif isinstance(dv_value, int):
            m = shank[:, 1] >= dv_value
        else:
            raise TypeError('')

        shank = shank[m]
        d = np.sqrt(np.sum((shank - shank[0]) ** 2, axis=1))

        return shank[d <= distance]

    @staticmethod
    def _shank_extend(shank: np.ndarray,
                      ext_depth: Optional[tuple[float, float]] = (0, 5000)) -> np.ndarray:
        """
        probe extension using extrapolation and interpolation

        :param shank: (2, 3), (points, (ap,dv,ml))
        :param ext_depth: depth in um, if None, only do the interpolation of the labelled points
        :return:
        """

        if ext_depth is not None:
            nn = np.arange(*ext_depth, 10)
        else:
            nn = np.arange(shank[0, 1], shank[-1, 1], 10)

        return interp1d(shank[:, 1], shank, axis=0, bounds_error=False, fill_value='extrapolate')(nn)


# ========= #


def _calc_shank_length_diff(shank: np.ndarray,
                            shank_interval: float):
    """
    use the vector of the probe, then calculate the unit vector

    :param shank: (N', 3)
    :param shank_interval: distance (um) relative to the specific shank. e.g., neuropixel2.0 = 250 * x
    :return:
    """
    v = shank[-1] - shank[0]
    v = v / np.linalg.norm(v)  # unit vector

    # vector product: |n x v| = sin_theta |n| |v|
    # inline n value, got following formula
    sin_theta = np.linalg.norm(np.array([v[2], 0, -v[0]]))

    return shank_interval * sin_theta


if __name__ == '__main__':
    ProbeReconstructor().main()
