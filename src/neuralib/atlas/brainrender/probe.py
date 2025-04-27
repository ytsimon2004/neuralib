from pathlib import Path

import numpy as np
import polars as pl
from brainglobe_atlasapi import BrainGlobeAtlas
from brainrender.actors import Points
from scipy.interpolate import interp1d
from typing_extensions import Self

from argclz import argument
from neuralib.atlas.brainrender.core import BrainRenderCLI
from neuralib.atlas.typing import PLANE_TYPE
from neuralib.atlas.util import allen_to_brainrender_coord
from neuralib.typing import PathLike

__all__ = ['ProbeRenderCLI',
           'ProbeShank']


class ProbeRenderCLI(BrainRenderCLI):
    """Probe track reconstruction with brainrender"""
    DESCRIPTION = 'Probe track reconstruction with brainrender'

    GROUP_PROBE = 'Probe Option'

    implant_depth: int = argument(
        '--depth',
        group=GROUP_PROBE,
        help='implant depth in um'
    )

    shank_interval: int | None = argument(
        '--interval',
        default=None,
        group=GROUP_PROBE,
        help='shank interval in um if multi-shank'
    )

    dye_label_only: bool = argument(
        '--dye',
        group=GROUP_PROBE,
        help='only show the histology dye parts'
    )

    remove_outside_brain: bool = argument(
        '--remove-outside-brain',
        group=GROUP_PROBE,
        help='remove reconstruction outside the brain'
    )

    file: Path = argument(
        '--file',
        metavar='FILE',
        type=Path,
        required=True,
        group=GROUP_PROBE,
        help='multi-shank npy or csv file to be inferred'
    )

    plane_type: PLANE_TYPE = argument(
        '--plane-type', '-P',
        default='coronal',
        group=GROUP_PROBE,
        help='cutting orientation to infer the multi-shank label point/probe_idx'
    )

    def post_parsing(self):
        super().post_parsing()
        if not self.dye_label_only and self.implant_depth is None:
            raise ValueError('')

    def run(self):
        if not self._stop_render:
            self.render()
            self.render_output()

    def render(self):
        self.post_parsing()
        super().render()
        self._add_probe()

    def _add_probe(self):
        bg = self.get_atlas_brain_globe()

        if self.file.suffix == '.csv':
            probe = ProbeShank.load_csv(self.file, self.plane_type, bg)
        elif self.file.suffix == '.npy':
            probe = ProbeShank.load_numpy(self.file, bg)
        else:
            raise TypeError(f"Unsupported file type: {self.file.suffix}")

        #
        if self.coordinate_space == 'ccf':
            probe = probe.map_brainrender()

        #
        if not self.dye_label_only:
            probe_theo = probe.as_theoretical(self.implant_depth, self.shank_interval, self.remove_outside_brain)
            self.scene.add(Points(probe_theo, colors='k', name='theo', alpha=0.9))

        probe_dye = probe.interp(ret_type=np.ndarray)
        self.scene.add(Points(probe_dye, colors='r', name='dye', alpha=0.9))


class ProbeShank:
    """shank reconstruction class"""

    def __init__(self, dorsal: np.ndarray,
                 ventral: np.ndarray,
                 bg: BrainGlobeAtlas):
        """

        :param dorsal: `Array[float, 3 | [S, 3]]`
        :param ventral: `Array[float, 3 | [S, 3]]`
        :param bg: ``BrainGlobeAtlas``
        """
        self._dorsal = dorsal
        self._ventral = ventral
        self._validate()

        self._bg = bg

    def _validate(self):
        assert self._dorsal.shape == self._ventral.shape

    def __iter__(self):
        """foreach shank DV"""
        for i in range(self.n_shanks):
            yield self._dorsal[i], self._ventral[i]

    @classmethod
    def load_numpy(cls, file: PathLike, bg: BrainGlobeAtlas) -> Self:
        """Load numpy array. `Array[float, [2, 3] | [S, 2, 3]]`

        ``S`` = Number of shanks. If 2D then single shank
        ``2`` = Dorsal and ventral
        ``3`` = AP, DV, ML coordinates
        """
        data = np.load(file)
        if data.ndim == 2:
            return cls(data[0], data[1], bg)
        elif data.ndim == 3:
            return cls(data[:, 0, :], data[:, 1, :], bg)
        else:
            raise ValueError(f'not support {data.shape=}')

    @classmethod
    def load_csv(cls, file: PathLike,
                 plane_type: PLANE_TYPE,
                 bg: BrainGlobeAtlas,
                 verbose: bool = True) -> Self:
        """
        Load from csv file

        :param file: csv file
        :param plane_type: {'coronal', 'sagittal', 'transverse'}
        :param bg: BrainGlobeAtlas
        :param verbose:
        :return:
        """
        df = pl.read_csv(file)
        cols = ['AP_location', 'DV_location', 'ML_location']
        if 'probe_idx' not in df.columns or 'point' not in df.columns:
            df = df.select(cols)
            df = cls._sort_rows(df, plane_type)

        if verbose:
            from neuralib.util.verbose import printdf
            printdf(df)

        data = df.select(cols).to_numpy().reshape(-1, 2, 3)
        return cls(data[:, 0, :], data[:, 1, :], bg)

    @classmethod
    def _sort_rows(cls, df: pl.DataFrame, plane_type: PLANE_TYPE) -> pl.DataFrame:
        n = int(df.shape[0] / 2)
        expr = pl.Series(['dorsal'] * n + ['ventral'] * n)
        df = df.sort('DV_location').with_columns(expr.alias('point'))

        if plane_type == 'sagittal':
            shank_order = 'AP_location'
        elif plane_type == 'coronal':
            shank_order = 'ML_location'
        else:
            raise ValueError('')

        df = (df.sort(by=['point', shank_order], descending=[False, True])
              .with_columns(pl.Series(list(range(1, 1 + n)) * 2).alias('probe_idx'))
              .sort('probe_idx'))

        return df

    @property
    def n_shanks(self) -> int:
        """number of shanks"""
        return self._dorsal.shape[0]

    def map_brainrender(self) -> Self:
        """map allen ccf brain space to brainrender"""
        self._dorsal = allen_to_brainrender_coord(self._dorsal)
        self._ventral = allen_to_brainrender_coord(self._ventral)
        return self

    def interp(self, interp_range: tuple[float, float] | None = None,
               ret_type: type = np.ndarray) -> list[np.ndarray] | np.ndarray:
        """extend_depth foreach shank

        :param interp_range:
        :param ret_type: if as list, then list[Array[float, [P, 3]]]. if numpy array `Array[float, [P * S, 3]]`
        :return: list[Array[float, [P, 3]]] | `Array[float, [P * S, 3]]`
        """
        ret = [self._interp(d, v, interp_range) for d, v in self]

        if ret_type is list:
            return ret
        elif ret_type is np.ndarray:
            return np.vstack(ret)
        else:
            raise ValueError('')

    def as_theoretical(self, depth: int, interval: int | None = None, remove_outside_brain: bool = True) -> np.ndarray:
        """
        as theoretical array

        :param depth: implanted depth
        :param interval: interval between shanks
        :param remove_outside_brain: remove the point outside the brain
        :return:
        """
        shanks = self.interp(interp_range=(0, 5000), ret_type=list)

        ret = []
        for shank in shanks:
            p = self._within_depth_range(shank, depth, remove_outside_brain)
            ret.append(p)

            if interval is not None:
                depth += self._calc_shank_length_diff(shanks[0], interval)
                interval += interval

        return np.vstack(ret)

    def _interp(self, d, v, interp_range: tuple[float, float] | None):
        """

        :param d: `Array[float, 3]`
        :param v: `Array[float, 3]`
        :param interp_range: interpolation DV in um. usually a large value, then cut afterward
        :return: `Array[float, [P, 3]]`
        """
        if interp_range is not None:
            nn = np.arange(*interp_range, 10)
        else:
            nn = np.arange(d[1], v[1], 10)

        s = np.vstack([d, v])

        return interp1d(s[:, 1], s, axis=0, bounds_error=False, fill_value='extrapolate')(nn)

    def _within_depth_range(self, shank: np.ndarray,
                            depth: int,
                            remove_outside_brain: bool = True):
        """remove point segments outside the brain, and more than given depth"""

        shank = shank[shank[:, 1] >= 0]

        if remove_outside_brain:
            mx = self._isin_brain(shank)
            shank = shank[mx]

        d = np.sqrt(np.sum((shank - shank[0]) ** 2, axis=1))

        return shank[d <= depth]

    def _isin_brain(self, shank: np.ndarray) -> np.ndarray:
        """

        :param shank: `Array[float, [P, 3]]`
        :param bg: ``BrainGlobeAtlas``
        :return: `Array[bool, P]`
        """
        bg = self._bg

        ret = []
        for p in shank:
            try:
                rid = bg.annotation[bg._idx_from_coords(p, microns=True)]
            except IndexError:
                rid = 0
            ret.append(rid != 0)

        return np.array(ret, dtype=bool)

    @staticmethod
    def _calc_shank_length_diff(shank: np.ndarray, shank_interval: float) -> float:
        """
        use the vector of the probe, then calculate the unit vector

        :param shank: `Array[float, [P, 3]]`
        :param shank_interval: distance (um) relative to the specific shank. e.g., NeuroPixel 2.0 = 250 * x
        :return: unit vector
        """
        v = shank[-1] - shank[0]
        v = v / np.linalg.norm(v)  # unit vector

        # vector product: |n x v| = sin_theta |n| |v|
        # inline n value, got following formula
        sin_theta = np.linalg.norm(np.array([v[2], 0, -v[0]]))

        return shank_interval * sin_theta


if __name__ == '__main__':
    ProbeRenderCLI().main()
