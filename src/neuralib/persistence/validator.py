import abc
import datetime
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np

from neuralib.typing import is_iterable
from neuralib.util.verbose import fprint
from .persistence import ensure_persistence_class

__all__ = [
    'create_date_validate',
    'attributes_validate',
    #
    'ETLConcatable',
    'validate_concat_etl_persistence',
]

T = TypeVar('T')


def _to_datetime(obj: float | Path | tuple[int, int, int] | datetime.date | datetime.datetime) -> datetime.datetime:
    if isinstance(obj, float):
        return datetime.datetime.fromtimestamp(obj)
    elif isinstance(obj, tuple):
        return datetime.datetime(obj[0], obj[1], obj[2])
    elif isinstance(obj, Path):
        if not obj.exists():
            raise FileNotFoundError
        return datetime.datetime.fromtimestamp(obj.stat().st_mtime)
    elif isinstance(obj, (datetime.datetime, datetime.date)):
        return obj
    else:
        raise TypeError(str(type(obj).__name__))


def create_date_validate(obj: float | Path | tuple[int, int, int] | datetime.date | datetime.datetime,
                         ref: float | Path | tuple[int, int, int] | datetime.date | datetime.datetime,
                         verbose=True) -> bool:
    """test *obj* is created after *ref*.

    :param obj: time stamp, file path or date instance
    :param ref: time stamp, file path or date instance
    :param verbose:
    :return: Is *obj* file newer than *ref* file?
    """
    try:
        obj_time = _to_datetime(obj)
    except FileNotFoundError:
        if verbose:
            fprint(f'obj file not existed : {obj}')
        return False

    try:
        ref_time = _to_datetime(ref)
    except FileNotFoundError:
        if verbose:
            fprint(f'ref file not existed : {ref}')
        return False

    if obj_time >= ref_time:
        return True

    if verbose:
        if isinstance(ref, float):
            msg = str(datetime.date.fromtimestamp(obj))
        elif isinstance(obj, tuple):
            msg = str(datetime.date(obj[0], obj[1], obj[2]))
        elif isinstance(ref, Path):
            msg = ref.name
        else:
            msg = str(ref)

        fprint(f'persistence data is created before {msg}')

    return False


def attributes_validate(obj: T, *exclude: str) -> bool:
    info = ensure_persistence_class(obj)
    for field in info.fields:
        attr = field.field_name
        if attr not in exclude and not hasattr(obj, attr) and not field.optional:
            return False
    return True


# ========================= #
# User-Specific for 2P data #
# ========================= #

class ETLConcatable(Generic[T], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def concat_etl(self, data: list[T]) -> T:
        pass


class PersistenceConcatError(Exception):
    pass


def validate_concat_etl_persistence(data: list[T], field_check: tuple[str, ...] | None = None) -> None:
    """

    :param data:
    :param field_check: field name under persistence cls for checking
    :return:
    """
    #
    if not hasattr(data[0], 'exp_date') or not hasattr(data[0], 'animal'):
        raise RuntimeError('different dataset is not concatable')

    #
    for it in data:

        if not hasattr(it, 'plane_index'):
            raise AttributeError('not plane index field')

        if not isinstance(it.plane_index, int):
            raise TypeError('')

    if len(set([it.exp_date for it in data])) != 1:
        print(set([it.exp_date for it in data]))
        raise PersistenceConcatError('different exp date')

    if len(set([it.animal for it in data])) != 1:
        raise PersistenceConcatError('different animal')

    #
    if field_check is not None:
        for f in field_check:
            init = getattr(data[0], f)

            for it in data:
                check = getattr(it, f)

                if not is_iterable(init):
                    if check != init:
                        raise PersistenceConcatError(f' field:{f} not consistent')

                elif is_iterable(init) and isinstance(init, np.ndarray):
                    if not np.array_equal(init, check):
                        raise PersistenceConcatError(f' field:{f} not consistent')

                else:
                    for i, v in enumerate(init):
                        check_field = check
                        if check_field[i] != v:
                            raise PersistenceConcatError(f' field:{f} not consistent')
