from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, overload

import numpy as np
import polars as pl
from polars import ColumnNotFoundError

from neuralib.stimpy import STIMPY_SOURCE_VERSION
from neuralib.util.util_verbose import fprint, printdf

__all__ = ['AbstractStimProtocol']


class AbstractStimProtocol(metaclass=abc.ABCMeta):
    """ABC class for the stimpy protocol file (.prot)

    `Dimension parameters`:

        N = numbers of visual stimulation (on-off pairs) = (T * S)

        T = number of trials

        S = Number of Stim Type

        C = Numbers of Cycle
    """

    name: str
    """protocol name. related to filename"""

    options: dict[str, Any]
    """protocol options"""

    visual_stimuli_dataframe: pl.DataFrame
    """visual stimuli dataframe. row number: S"""

    version: STIMPY_SOURCE_VERSION
    """date of major changes"""

    def __repr__(self):
        ret = list()

        ret.append('# general parameters')
        for k, v in self.options.items():
            ret.append(f'{k} = {v}')
        ret.append('# stimulus conditions')
        ret.append('\t'.join(self.stim_params))

        ret.append(printdf(self.visual_stimuli_dataframe, do_print=False))

        return '\n'.join(ret)

    def __init__(self, name: str,
                 options: dict[str, Any],
                 visual_stimuli: pl.DataFrame,
                 version: STIMPY_SOURCE_VERSION):

        self.name = name
        self.options = options
        self.visual_stimuli_dataframe = visual_stimuli
        self.version = version

        self._visual_stimuli = None

    @property
    def n_stimuli(self) -> int:
        """number of stimuli (S)"""
        return self.visual_stimuli_dataframe.shape[0]

    @property
    def stim_params(self) -> tuple[str, ...]:
        return tuple(self.visual_stimuli_dataframe.columns)

    # @property
    # def visual_stimuli(self) -> list[GenericStim]:  # TODO check
    #     if self._visual_stimuli is None:
    #         from rscvp.ztimpy.stim.stimulus import VisualStimulus
    #         self._visual_stimuli = VisualStimulus.from_protocol(self)
    #     return self._visual_stimuli

    @overload
    def __getitem__(self, item: int) -> pl.DataFrame:
        """get row of visual stimuli"""
        pass

    @overload
    def __getitem__(self, item: str) -> np.ndarray:
        """get header of stimuli"""
        pass

    def __getitem__(self, item: str | int) -> np.ndarray | pl.DataFrame:
        """Get protocol value of parameter *item*

        :param item: parameter name. either the row number(int) or field name (str)
        :return: protocol value
        :raises TypeError: `item` not a `str` or `int`
        """
        if isinstance(item, str):
            try:
                return self.visual_stimuli_dataframe.get_column(item).to_numpy()
            except ColumnNotFoundError as e:
                fprint(f'INVALID: {item}, select from {tuple(self.visual_stimuli_dataframe.columns)}', vtype='error')
                raise e

        elif isinstance(item, int):
            ret = {
                h: self.visual_stimuli_dataframe.row(item)[i]
                for i, h in enumerate(self.visual_stimuli_dataframe.columns)
            }
            return pl.DataFrame(ret)
        else:
            raise TypeError(f'{type(item)}')

    @classmethod
    @abc.abstractmethod
    def load(cls, file: Path | str) -> AbstractStimProtocol:
        """Load *.prot file

        :param file: file path
        """
        pass

    @property
    @abc.abstractmethod
    def is_shuffle(self) -> bool:
        """if shuffle stimulation"""
        pass

    @property
    @abc.abstractmethod
    def background(self) -> float:
        """background for non-stimulation epoch.
        Note the default value need to check in user-specific stimpy version
        """
        pass

    @property
    @abc.abstractmethod
    def start_blank_duration(self) -> int:
        """blank duration before starting the visual stimulation epoch (in sec)"""
        pass

    @property
    @abc.abstractmethod
    def blank_duration(self) -> int:
        """blank duration between each visual stimulus (in sec)"""
        pass

    @property
    @abc.abstractmethod
    def trial_blank_duration(self) -> int:
        """blank duration between trials (in sec)
        TODO check stimpy source code"""
        pass

    @property
    @abc.abstractmethod
    def end_blank_duration(self) -> int:
        """blank duration after the visual stimulation epoch"""
        pass

    @property
    @abc.abstractmethod
    def trial_duration(self) -> int:
        """trial duration"""
        pass

    @property
    def visual_duration(self) -> int:
        """total visual duration"""
        return self.trial_duration * self.n_trials

    @property
    def total_duration(self) -> int:
        """total protocol duration"""
        return self.start_blank_duration + self.visual_duration + self.end_blank_duration

    @property
    def stimulus_type(self) -> str:
        """stimulus type"""
        return self.options['stimulusType']

    @property
    def n_trials(self) -> int:
        """(T,)"""
        return self.options['nTrials']

    def to_dict(self) -> dict[str, Any]:
        """to stimpy runtime readable dict TODO"""
        pass
