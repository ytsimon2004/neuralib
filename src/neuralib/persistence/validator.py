import abc
from typing import Generic, TypeVar

import numpy as np

from neuralib.typing import is_iterable

__all__ = [
    'ETLConcatable',
    'validate_concat_etl_persistence',
]

T = TypeVar('T')


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
