from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import TYPE_CHECKING

from neurolib.stimpy.baselog import Baselog, LOG_SUFFIX
from neurolib.stimpy.baseprot import AbstractStimProtocol
from neurolib.stimpy.util import try_casting_number, unfold_stimuli_condition
from neurolib.util.util_type import PathLike
from neurolib.util.util_verbose import fprint

if TYPE_CHECKING:
    from rscvp.util.stimlog_pyvstim import StimlogPyVStim

import numpy as np
import polars as pl

__all__ = ['PyVlog',
           'PyVProtocol']


class PyVlog(Baselog):
    """PyVStim log parsing"""

    def __init__(self,
                 root_path: PathLike,
                 log_suffix: LOG_SUFFIX = '.log',
                 diode_offset: bool = False):

        super().__init__(root_path, log_suffix, diode_offset)
        self.__prot_cache: PyVProtocol | None = None

    @classmethod
    def _cache_asarray(cls, filepath: Path) -> np.ndarray:
        output = filepath.with_name(filepath.stem + '_log.npy')

        if not output.exists():

            data_list = []
            with filepath.open() as f:
                for line, content in enumerate(f):
                    content = content.strip()
                    if not content.startswith('#') and content != '':  # comments and empty line
                        cols = content.strip().split(',')
                        # Convert the columns to floats
                        cols = [float(x) for x in cols]
                        # Append the row to data_list
                        data_list.append(cols)

            # Find the maximum number of columns
            max_cols = max([len(row) for row in data_list])

            new_data = []

            # Iterate over each row
            for row in data_list:
                # Calculate the number of columns to add
                cols_to_add = max_cols - len(row)
                # Add the required number of np.nan values
                row.extend([np.nan] * cols_to_add)
                # Append the row to new_data
                new_data.append(row)

            # Convert new_data to a numpy array
            ret = np.array(new_data)

            np.save(output, ret)

        return np.load(output)

    # ===== #

    def stimlog_data(self) -> 'StimlogPyVStim':
        from rscvp.util.stimlog_pyvstim import StimlogPyVStim
        return StimlogPyVStim(self)

    def get_prot_file(self) -> PyVProtocol:
        if self.__prot_cache is None:
            self.__prot_cache = PyVProtocol.load(self.stim_prot_file)

        return self.__prot_cache


# ======== #
# Protocol #
# ======== #

class PyVProtocol(AbstractStimProtocol):

    @classmethod
    def load(cls, file: Path | str, *,
             cast_numerical_opt=True) -> 'PyVProtocol':

        file = Path(file)
        options = {}
        version = 'pyvstim'

        state = 0
        with Path(file).open() as f:
            for line in f:
                line = line.strip()

                if len(line) == 0 or line.startswith('#'):
                    continue

                if state == 0 and line.startswith('n\t'):
                    header = re.split(r'\t+| +', line)
                    value = [[] for _ in range(len(header))]
                    state = 1

                elif state == 0:
                    idx = line.index('=')
                    if cast_numerical_opt:
                        opt_value = try_casting_number(line[idx + 1:].strip())
                    else:
                        opt_value = line[idx + 1:].strip()

                    options[line[:idx].strip()] = opt_value

                elif state == 1:
                    parts = re.split(r'\t+| +', line, maxsplit=len(header))
                    rows = unfold_stimuli_condition(parts)

                    for r in rows:
                        r.remove('')  # pyvstim interesting problem
                        for i, it in enumerate(r):  # for each col
                            if it != '':
                                value[i].append(it)
                else:
                    raise RuntimeError('illegal state')

            assert len(header) == len(value)
            visual_stimuli = {
                field: value[i]
                for i, field in enumerate(header)
            }

            if 'Shuffle' not in options.keys():
                options['Shuffle'] = False

        return PyVProtocol(file.name, options, pl.DataFrame(visual_stimuli), version)

    @property
    def shuffle(self) -> bool:
        """TODO"""
        return False

    @property
    def background(self) -> float:
        """TODO"""
        return self.options.get('background', 0.5)

    @property
    def start_blank_duration(self) -> int:
        raise NotImplementedError('')

    @property
    def blank_duration(self) -> int:
        return self.options['BlankDuration']

    @property
    def trial_blank_duration(self) -> int:
        raise NotImplementedError('')

    @property
    def end_blank_duration(self) -> int:
        raise NotImplementedError('')

    @property
    def trial_duration(self) -> int:
        raise NotImplementedError('')

    @property
    def visual_duration(self) -> int:
        raise NotImplementedError('')

    @property
    def total_duration(self) -> int:
        raise NotImplementedError('')

    def get_loops_expr(self) -> ProtExpression:
        """parse and get the expression and loop number"""
        exprs = []
        n_cycles = []
        n_blocks = self.visual_stimuli_dataframe.shape[0]

        for row in self.visual_stimuli_dataframe.iter_rows():  # each row item_value
            for it in row:
                if isinstance(it, str):
                    if 'loop' in it:
                        match = re.search(r"loop\((.*),(\d+)\)", it)

                        if match:
                            exprs.append(match.group(1))
                            n_cycles.append(match.group(2))
                    else:
                        fprint('loop info not found, check prot file!', vtype='warning')
                        exprs.append('')
                        n_cycles.append(1)

        return ProtExpression(exprs, list(map(int, n_cycles)), n_blocks)


@dataclasses.dataclass
class ProtExpression:
    expr: list[str]
    """expression"""
    n_cycles: list[int]
    """number of cycle. len:"""
    n_blocks: int | None
    """number of prot value row (block)"""

    def __post_init__(self):
        if (len(self.n_cycles) == 2 * self.n_blocks) and self._check_ncycles_foreach_block():
            self.n_cycles = self.n_cycles[::2]
        else:
            raise RuntimeError('')

    def _check_ncycles_foreach_block(self):
        """check if the ncycles are the same and duplicate for each block"""
        n = len(self.n_cycles)
        if n % 2 != 0:
            return False

        for i in range(0, n, 2):
            if self.n_cycles[i] != self.n_cycles[i + 1]:
                return False

        return True
