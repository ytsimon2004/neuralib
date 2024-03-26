from __future__ import annotations

from datetime import datetime
from typing import Literal

import polars as pl
import pandas as pd
from colorama import Fore, Style

from neurolib.util.util_type import DataFrame

__all__ = ['fprint',
           'printdf']


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
            do_fprint=False,
            **kwargs) -> str:
    """print polars dataframe with given row numbers"""

    if isinstance(df, pl.DataFrame):
        with pl.Config() as cfg:
            rows = df.shape[0] if nrows is None else nrows
            cols = df.shape[1] if ncols is None else ncols
            cfg.set_tbl_rows(rows)
            cfg.set_tbl_cols(cols)

            if do_fprint:
                fprint(df, **kwargs)
            else:
                print(df)

            return df.__repr__()

    elif isinstance(df, pd.DataFrame):
        ret = df.to_markdown()
        print(ret)
        return ret

    else:
        raise TypeError('')
