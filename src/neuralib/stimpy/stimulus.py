from __future__ import annotations

from typing import NamedTuple, Iterable, TYPE_CHECKING

import numpy as np
from typing_extensions import TypeAlias, Self

if TYPE_CHECKING:
    from neuralib.stimpy.baselog import R

__all__ = [
    'Degree',
    'SF',
    'TF',
    'SFTF',
    'VisualParas',
    #
    'StimPattern'
]

Degree: TypeAlias = int  # degree
SF: TypeAlias = float  # cyc/deg
TF: TypeAlias = int  # Hz
SFTF: TypeAlias = tuple[SF, TF]
VisualParas: TypeAlias = tuple[SF, TF, Degree]


class StimPattern(NamedTuple):
    """Stimulus Parameters

    `Dimension parameters`:

        N = numbers of visual stimulation (on-off pairs) = (T * S)
    """
    time: np.ndarray
    """stim on-off in sec (N, 2)"""
    direction: np.ndarray  # int (degree) (N,)
    """degree (N,)"""
    sf: np.ndarray  # float (cyc/deg) (N,)
    """cyc/deg (N,)"""
    tf: np.ndarray  # int (Hz) (N,)
    """hz (N,)"""
    contrast: np.ndarray
    """stimulus contrast (N,)"""
    duration: np.ndarray
    """theoretical duration in prot file, not actual detected using diode"""

    @classmethod
    def of(cls, rig: 'R') -> Self:
        """
        init from Baselog children class

        :param rig: :class:`~neuralib.stimpy.baselog.Baselog`
        :return: :class:`StimPattern`
        """
        return rig.stimlog_data().get_stim_pattern()

    @property
    def sf_set(self) -> np.ndarray:
        """unique sf_set"""
        return np.array(sorted(self.sf_i().keys()))

    @property
    def tf_set(self) -> np.ndarray:
        """unique tf_set"""
        return np.array(sorted(self.tf_i().keys()))

    @property
    def n_sf(self) -> int:
        """number of sf set"""
        return len(self.sf_set)

    @property
    def n_tf(self) -> int:
        """number of tf set"""
        return len(self.tf_set)

    @property
    def n_sftf(self) -> int:
        """number of sftf combination"""
        return len(self.sftf_i())

    @property
    def n_dir(self) -> int:
        """number of direction"""
        return len(self.dir_i())

    def dir_i(self) -> dict[Degree, int]:
        """deg:index dict"""
        return {it: i for i, it in enumerate(sorted(np.unique(self.direction)))}

    def sf_i(self) -> dict[SF, int]:
        """sf:index dict"""
        return {it: i for i, it in enumerate(sorted(np.unique(self.sf)))}

    def tf_i(self) -> dict[TF, int]:
        """sf:index dict"""
        return {it: i for i, it in enumerate(sorted(np.unique(self.tf)))}

    # previous plot use tfsf as condition idx
    def sftf_i(self) -> dict[SFTF, int]:
        """sf, tf combination. (sf , tf):y"""
        return {
            it: i
            for i, it in enumerate([
                (sf, tf)
                for sf in sorted(np.unique(self.sf))
                for tf in sorted(np.unique(self.tf))
            ])
        }

    def sftfdir_i(self) -> dict[VisualParas, int]:
        """sf, tf, ori combination. (sf , tf , ori):y"""
        return {
            it: i  # (sf , tf, ori): index
            for i, it in enumerate([
                (sf, tf, ori * 30)
                for sf in sorted(np.unique(self.sf))
                for tf in sorted(np.unique(self.tf))
                for ori in range(12)
            ])
        }

    def foreach_stimulus(self) -> Iterable[tuple[int, np.ndarray, SF, TF, Degree]]:
        """Generator for (index, stimulus_time, sf, tf, ori)"""
        for si, st in enumerate(self.time):
            yield si, st, self.sf[si], self.tf[si], self.direction[si]

    def get_stim_time(self) -> float:
        """get approximate stim time if the same duration. i.e., for plotting purpose"""
        return np.mean(self.time[:, 1] - self.time[:, 0])
