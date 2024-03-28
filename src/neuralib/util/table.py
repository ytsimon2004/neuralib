import contextlib
import functools
import textwrap
from typing import Union, ContextManager, Dict, Optional, List, Tuple

import pandas as pd
import polars as pl
import rich
from rich import box
from rich.console import Console, Capture
from rich.table import Table, Column

__all__ = [
    'Column',
    'Table',
    'rich_table',
    'rich_table_content',
    'rich_default_table',
    'rich_data_frame_table',
    'show_polars_header_types'
]


class TableLike:
    def __init__(self, table: Table, capture: Optional[Capture] = None):
        self.__table = table
        self.__capture = capture

    def add_column(self, header=None, footer=None, **kwargs):
        self.__table.add_column(header, footer, **kwargs)

    def __call__(self, *args):
        """

        :rtype: object
        """
        self.__table.add_row(*[str(it) if it is not None else '' for it in args])

    def get(self, prepend: Union[int, str] = None):
        ret = self.__capture.get()

        if prepend is not None:
            if isinstance(prepend, int):
                prepend = ' ' * prepend
            ret = textwrap.indent(ret, prepend)

        return ret


def _rich_table_header(*header: Union[int, str, Column]) -> List[Column]:
    def _header(ih: Tuple[int, Union[int, str, Column]]) -> Column:
        i, h = ih
        if isinstance(h, int):
            if i + 1 == len(header):  # last one
                return Column('', max_width=h)
            else:
                return Column('', min_width=h)
        elif isinstance(h, str):
            return Column(h)
        elif isinstance(h, Column):
            return h
        else:
            raise TypeError()

    return list(map(_header, enumerate(header)))


@functools.wraps(Table.__init__)
def rich_default_table(*header: Union[str, Column], **kwargs) -> Table:
    kwargs.setdefault('show_edge', False)
    kwargs.setdefault('box', box.SIMPLE)
    return Table(*_rich_table_header(*header), **kwargs)


@contextlib.contextmanager
@functools.wraps(Table.__init__)
def rich_table(*header: Union[int, str, Column],
               **kwargs) -> ContextManager[TableLike]:
    table = rich_default_table(*header, **kwargs)
    yield TableLike(table)
    rich.get_console().print(table)


@contextlib.contextmanager
@functools.wraps(Table.__init__)
def rich_table_content(*header: Union[int, str, Column],
                       **kwargs) -> ContextManager[TableLike]:
    console = Console()
    with console.capture() as capture:
        table = rich_default_table(*header, **kwargs)
        yield TableLike(table, capture)
        console.print(table)


def rich_data_frame_table(frame: Union[pd.DataFrame, pl.DataFrame, Dict], *,
                          show_dytpe=False) -> Table:
    if isinstance(frame, pd.DataFrame):
        return _rich_pandas_data_frame_table(frame, show_dytpe=show_dytpe)
    elif isinstance(frame, pl.DataFrame):
        return _rich_polars_data_frame_table(frame, show_dytpe=show_dytpe)
    else:
        return _rich_polars_data_frame_table(pl.DataFrame(frame), show_dytpe=show_dytpe)


def _rich_pandas_data_frame_table(frame: pd.DataFrame, *,
                                  show_dytpe=False) -> Table:
    table = rich_default_table()
    table.add_column(frame.index.name or '')
    for c in frame.columns:
        table.add_column(c)

    if show_dytpe:
        # TODO
        pass

    for c, d in frame.iterrows():
        table.add_row(str(c), *list(map(str, d)))

    rich.get_console().print(table)
    return table


def _rich_polars_data_frame_table(frame: pl.DataFrame, *,
                                  show_dytpe=False) -> Table:
    table = rich_default_table()
    for c in frame.columns:
        table.add_column(c)

    if show_dytpe:
        table.add_row(*list(map(str, frame.dtypes)))

    for row in frame.iter_rows():
        table.add_row(*list(map(str, row)))

    rich.get_console().print(table)
    return table


def show_polars_header_types(frame: pl.DataFrame):
    with pl.Config() as cfg:
        cfg.set_tbl_rows(0)
        cfg.set_tbl_hide_column_data_types(False)
        cfg.set_tbl_hide_column_names(False)
        cfg.set_tbl_hide_dtype_separator(True)
        cfg.set_tbl_hide_dataframe_shape(True)
        cfg.set_tbl_cols(-1)  # display all
        print(frame)
