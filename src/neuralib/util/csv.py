from __future__ import annotations

from pathlib import Path
from typing import Optional, Set

__all__ = ['csv_header']


class CsvContextManager:
    def __init__(self,
                 output: Optional[Path],
                 header: list[str],
                 append=False,
                 buffer=False,
                 continuous_mode: str = None,
                 quotes_header: list[str] = None):
        self._output = output
        self._header = header
        self._append = append
        self._buffer = buffer
        self._cont_mode = continuous_mode

        self._stream = None
        self._cont_column: Optional[Set[str]] = None
        self._cont_index = -1
        self._quotes_header: Optional[list[str]] = quotes_header

    def __enter__(self):
        if self._stream is not None:
            raise RuntimeError()

        if self._quotes_header is not None:
            if not all([it in self._header for it in self._quotes_header]):
                raise RuntimeError('')

        self._cont_column = None
        self._cont_index = -1

        if self._output is not None:
            if self._cont_mode is not None and self._cont_mode in self._header and self._append:
                self._cont_column = set()
                self._cont_index = self._header.index(self._cont_mode)
                if self._output.exists():
                    with self._output.open('r') as f:
                        for line in f:
                            try:
                                p = line.strip().split(',')[self._cont_index]
                            except IndexError:
                                pass
                            else:
                                self._cont_column.add(p)

            print_header = not self._output.exists() or not self._append
            if self._buffer:
                from io import StringIO
                self._stream = StringIO()
            else:
                mode = 'a' if self._append else 'w'
                self._stream = self._output.open(mode)

            if print_header:
                print(*self._header, sep=',', file=self._stream, flush=True)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._output is None:
            return

        if self._buffer:
            if exc_type is None:
                mode = 'a' if self._append else 'w'
                with self._output.open(mode) as out:
                    out.write(self._stream.getvalue())
            else:
                print('error happen, drop buffer')

        self._stream.close()

        self._stream = None

    def __call__(self, *args):
        if self._output is None:
            return

        if self._quotes_header is not None:
            ret = list([*args])
            header_idx = [self._header.index(q) for q in self._quotes_header]
            for i in header_idx:
                ret[i] = f'\"{ret[i]}\"'

            args = tuple(ret)

        print(*args, sep=',', file=self._stream, flush=True)
        if self._cont_column is not None:
            self._cont_column.add(str(args[self._cont_index]))

    def __contains__(self, item):
        if self._cont_column is None:
            return False

        return str(item) in self._cont_column

    def __repr__(self):
        return str(self._output)


def csv_header(output: Optional[Path],
               header: list[str],
               append: bool = False,
               buffer: bool = False,
               continuous_mode: str = None,
               quotes_header: list[str] | str = None) -> CsvContextManager:
    """

    **Example**

    >>> csv_output = Path('output.csv')
    >>> with csv_header(csv_output, ['neuron_id', 'data']) as csv:
    ...     for neuron in range(total_neuron):
    ...         csv(neuron, neuron + 1)

    generate file output.csv with content::

        neuron_id,data
        0,1
        1,2
        2,3

    continuous mode

    >>> with csv_header(csv_output, ['neuron_id', 'data'], append=True, continuous_mode='neuron_id') as csv:
    ...     for neuron in range(total_neuron):
    ...         if neuron not in csv:
    ...             csv(neuron, neuron + 1)



    :param output: output path
    :param header: csv header list
    :param append: if true, append in a same output csv file next time. otherwise, overwrite
    :param buffer: buffering output if append mode
    :param continuous_mode: continuous append csv file with append mode. Give a field name which must be a member
        in the header.
    :param quotes_header: list of header that need to use double quotes

    :return: row data consumer.
    """
    if isinstance(quotes_header, str):
        quotes_header = [quotes_header]

    return CsvContextManager(output, header, append, buffer, continuous_mode, quotes_header)
