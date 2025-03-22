from datetime import datetime
from pathlib import Path
from typing import Literal, Callable

import h5py
import pandas as pd
import polars as pl
from colorama import Fore, Style
from neuralib.typing import DataFrame, PathLike

__all__ = ['fprint',
           'printdf',
           'print_h5py',
           'print_load',
           'print_save',
           'publish_annotation']

_PREV_LOAD_FILE = None
_PREV_SAVE_FILE = None


def fprint(*msgs,
           vtype: Literal['info', 'io', 'warning', 'error', 'pass', 'debug'] | str = 'info',
           timestamp: bool = True,
           **kwarg) -> None:
    """
    Formatting print with different colors based on verbose type

    :param msgs: print message
    :param vtype: verbose type
    :param timestamp: If add time info
    """
    prefix = f'[{vtype.upper()}]'

    if vtype == 'error':
        color = 'RED'
    elif vtype == 'warning':
        color = 'YELLOW'
    elif vtype == 'io':
        color = 'MAGENTA'
    elif vtype == 'info':
        color = 'CYAN'
    elif vtype == 'pass':
        color = 'GREEN'
    else:
        color = 'WHITE'

    try:
        fg_color = getattr(Fore, color)
    except AttributeError:
        fg_color = Fore.WHITE

    msg = fg_color + prefix
    if timestamp:
        msg += f"[{datetime.today().strftime('%y-%m-%d %H:%M:%S')}] - "

    try:
        out = ''.join(msgs) + "\n"
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


def print_h5py(group: h5py.Group | PathLike, indent: int = 0) -> None:
    if isinstance(group, PathLike):
        group = h5py.File(group)

    for key in group:
        item = group[key]
        prefix = " " * indent
        if isinstance(item, h5py.Group):
            print(f"{prefix}Group: {key}")
            print_h5py(item, indent=indent + 4)
        elif isinstance(item, h5py.Dataset):
            print(f"{prefix}Dataset: {key} (shape: {item.shape}, dtype: {item.dtype})")
        else:
            print(f"{prefix}{key}: Unknown type {type(item)}")


def print_load(file: PathLike, verb='LOAD') -> Path:
    """print message for loading file.

    If *file* is not existed, '!' will add after *verb*.
    File load message doesn't print twice and more if it shows continuously, so it is okay
    that wrap a loading method and call :func:`print_load` on target file.

    use flag "load".

    :param file: file path
    :param verb: default 'LOAD'
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


def print_save(file: PathLike, verb='SAVE') -> Path:
    """print message for saving file.

    If *file* is not existed, '+' will add after *verb*.
    File save message doesn't print twice and more if it shows continuously, so it is okay
    that wrap a saving method and call :func:`print_load` on target file.

    use flag "save".

    :param file: file path
    :param verb: default 'SAVE'
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


def publish_annotation(level: Literal['main', 'sup', 'appendix', 'test'],
                       *,
                       project: str | list[str] | None = None,
                       figure: str | list[str] | None = None,
                       caption: str | None = None,
                       as_doc: bool = False,
                       as_attributes: bool = True):
    """
    Annotation for knowing the class/function using scenario in paper publication

    :param level: {'main', 'sup', 'appendix', 'test'}
    :param project: Project name or list of project name
    :param figure: Figure number or list of figure name
    :param caption: Other caption
    :param as_doc: As documents, be able to parser the `.. note::` block by ``Sphinx``
    :param as_attributes: If set info as attributes
    :return:
    """
    valid_levels = ('main', 'sup', 'appendix', 'test')
    if level not in valid_levels:
        raise ValueError(f'must be one of the {valid_levels}')

    def decorator(target: type | Callable):

        doc = '' if target.__doc__ is None else target.__doc__

        if as_doc:
            doc += '\n\n' + '\n\n'.join([
                '.. note:: ',
                '\t**Publish Annotation**',
                f'\tProject: {project}',
                f'\tFigure: {figure}',
                f'\tLevel: {level} ',
                f'\tCaption: {caption}',
            ])
            target.__doc__ = doc

        if as_attributes:
            attrs = ['__publish_level__', '__publish_project__', '__publish_figure__', '__publish_caption__']

            for attr in attrs:
                if hasattr(target, attr):
                    raise AttributeError(f"Class {target.__name__} already has an attribute named '{attr}'.")

            target.__publish_level__ = level
            target.__publish_project__ = project
            target.__publish_figure__ = figure
            target.__publish_caption__ = caption

        return target

    return decorator
