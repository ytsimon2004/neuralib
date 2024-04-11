from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Union

import pandas as pd
import polars as pl
from colorama import Fore, Style

from neuralib.util.util_type import DataFrame

__all__ = ['fprint',
           'printdf',
           'print_load',
           'print_save']

_PREV_LOAD_FILE = None
_PREV_SAVE_FILE = None


def fprint(*msgs,
           vtype: Literal['info', 'io', 'warning', 'error', 'pass'] = 'info',
           timestamp: bool = True,
           **kwarg) -> None:
    """
    Formatting print with different colors based on verbose type

    :param msgs:
    :param vtype: verbose type
    :param timestamp:
    :return:
    """

    if vtype == 'error':
        prefix = '[ERROR]'
        color = 'red'
    elif vtype == 'warning':
        prefix = '[WARNING] '
        color = 'yellow'
    elif vtype == 'io':
        prefix = '[IO] '
        color = 'magenta'
    elif vtype == 'info':
        prefix = '[INFO]'
        color = 'cyan'
    elif vtype == 'pass':
        prefix = '[PASS]'
        color = 'green'
    else:
        raise ValueError(f'{vtype}')

    try:
        fg_color = getattr(Fore, color.upper())
    except AttributeError:
        fg_color = Fore.WHITE

    msg = fg_color + prefix
    if timestamp:
        msg += f"[{datetime.today().strftime('%y-%m-%d %H:%M:%S')}] - "

    try:
        out = f"{''.join(msgs)}\n"
    except TypeError:
        out = f'{msgs}'

    msg += out
    msg += Style.RESET_ALL
    print(msg, **kwarg)


def printdf(df: DataFrame,
            nrows: int | None = None,
            ncols: int | None = None,
            tbl_width_chars: int = 500,
            do_print: bool = True) -> str:
    """
    print dataframe with given row numbers (polars)
    if isinstance pandas dataframe, print all.

    :param df: polars or pandas dataframe
    :param nrows: number of rows (applicable in polars case)
    :param ncols: number of columns
    :param tbl_width_chars: table width for showing
    :param do_print: do print otherwise, only return the str
    :return:
    """

    if isinstance(df, pl.DataFrame):
        with pl.Config(tbl_width_chars=tbl_width_chars) as cfg:
            rows = df.shape[0] if nrows is None else nrows
            cols = df.shape[1] if ncols is None else ncols
            cfg.set_tbl_rows(rows)
            cfg.set_tbl_cols(cols)

            if do_print:
                print(df)

            return df.__repr__()

    elif isinstance(df, pd.DataFrame):
        ret = df.to_markdown()
        print(ret)
        return ret

    else:
        raise TypeError('')


def print_load(file: Union[str, Path], verb='LOAD') -> Path:
    """print message for loading file.

    If *file* is not existed, '!' will add after *verb*.
    File load message doesn't print twice and more if it shows continuously, so it is okay
    that wrap a loading method and call :func:`print_load` on target file.

    use flag "load".

    :param file: file path
    :param verb: default 'LOAD'
    :param color: verb color
    :return: *file*
    """
    global _PREV_LOAD_FILE

    file = Path(file)

    if _PREV_LOAD_FILE != file:
        if len(verb) < 4:
            verb += ' ' * (4 - len(verb))

        not_exist = '!' if not file.exists() else ' '

        from tqdm import tqdm
        with tqdm.external_write_mode():
            fprint(f'{verb}{not_exist}->{file}', vtype='io')

    _PREV_LOAD_FILE = file
    return file


def print_save(file: Union[str, Path], verb='SAVE') -> Path:
    """print message for saving file.

    If *file* is not existed, '+' will add after *verb*.
    File save message doesn't print twice and more if it shows continuously, so it is okay
    that wrap a saving method and call :func:`print_load` on target file.

    use flag "save".

    :param file: file path
    :param verb: default 'SAVE'
    :param color: verb color
    :return: *file*
    """
    global _PREV_SAVE_FILE

    file = Path(file)

    if _PREV_SAVE_FILE != file:
        if len(verb) < 4:
            verb += ' ' * (4 - len(verb))

        not_exist = '+' if not file.exists() else ' '

        from tqdm import tqdm
        with tqdm.external_write_mode():
            fprint(f'{verb}{not_exist}->{file}', vtype='io')

    _PREV_SAVE_FILE = file
    return file
