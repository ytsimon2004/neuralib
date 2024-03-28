from __future__ import annotations

from pathlib import Path
from typing import Union

from neuralib.util.util_verbose import fprint

_PREV_LOAD_FILE = None
_PREV_SAVE_FILE = None

__all__ = ['print_load',
           'print_save']


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
