from typing import TypeVar, Callable, Union

__all__ = [
    'try_int_type',
    'int_tuple_type',
    'str_tuple_type',
    'float_tuple_type',
    'tuple_type',
    'union_type',
    'dict_type',
    'slice_type'
]

T = TypeVar('T')


def tuple_type(value_type: Callable[[str], T]):
    def _type(arg: str) -> tuple[T, ...]:
        return tuple(map(value_type, arg.split(',')))

    return _type


str_tuple_type = tuple_type(str)
int_tuple_type = tuple_type(int)
float_tuple_type = tuple_type(float)


def union_type(*t):
    def _type(arg: str):
        for _t in t:
            try:
                return _t(arg)
            except (TypeError, ValueError):
                pass
        raise TypeError

    return _type


def dict_type(default: dict[str, T], value_type: Callable[[str], T] = None):
    """Dict arg value.

    :param default: default dict content
    :param value_type: type of dict value
    :return: type converter
    """
    if default is None:
        default = {}

    def _type(arg: str) -> dict[str, T]:
        if ':' in arg:
            i = arg.index(':')
            value = arg[i + 1:]
            if value_type is not None:
                value = value_type(value)
            default[arg[:i]] = value
        elif '=' in arg:
            i = arg.index('=')
            value = arg[i + 1:]
            if value_type is not None:
                value = value_type(value)
            default[arg[:i]] = value
        elif value_type is None:
            default[arg] = None
        else:
            default[arg] = value_type("")
        return default

    return _type


def slice_type(arg: str) -> slice:
    i = arg.index(':')
    v1 = int(arg[:i])
    v2 = int(arg[i + 1:])
    return slice(v1, v2)


def try_int_type(arg: str) -> Union[int, str, None]:
    """for argparse (i.e., plane_index)"""
    if len(arg) == 0:
        return None
    try:
        return int(arg)
    except ValueError:
        return arg
