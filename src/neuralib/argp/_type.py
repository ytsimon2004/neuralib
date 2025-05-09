from typing import TypeVar, Callable, Union, overload, Literal, get_origin, get_args

from neuralib.util.deprecation import deprecated_func

__all__ = [
    'try_int_type',
    'try_float_type',
    'int_tuple_type',
    'str_tuple_type',
    'float_tuple_type',
    'tuple_type',
    'list_type',
    'union_type',
    'dict_type',
    'slice_type'
]

T = TypeVar('T')


@deprecated_func(remarks='use the independent package: `pip install argclz`', removal_version='0.5.0')
def tuple_type(value_type: Callable[[str], T]):
    def _type(arg: str) -> tuple[T, ...]:
        return tuple(map(value_type, arg.split(',')))

    return _type


@deprecated_func(remarks='use the independent package: `pip install argclz`', removal_version='0.5.0')
def list_type(value_type: Callable[[str], T] = str, *, split=',', prepend: list[T] = None):
    """:attr:`arg.type` caster which convert comma ',' spread string into list.

    :param split: split character
    :param value_type: value type converter
    :param prepend: prepend list
    :return: type caster.
    """

    def _cast(arg: str) -> list[T]:
        value = list(map(value_type, arg.split(split)))

        if arg.startswith('+') and prepend is not None:
            return [*prepend, *value]
        else:
            return list(value)

    return _cast


str_tuple_type = tuple_type(str)
int_tuple_type = tuple_type(int)
float_tuple_type = tuple_type(float)


@deprecated_func(remarks='use the independent package: `pip install argclz`', removal_version='0.5.0')
def union_type(*t):
    def _type(arg: str):
        for _t in t:
            try:
                return _t(arg)
            except (TypeError, ValueError):
                pass
        raise TypeError

    return _type


@deprecated_func(remarks='use the independent package: `pip install argclz`', removal_version='0.5.0')
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


@deprecated_func(remarks='use the independent package: `pip install argclz`', removal_version='0.5.0')
def slice_type(arg: str) -> slice:
    i = arg.index(':')
    v1 = int(arg[:i])
    v2 = int(arg[i + 1:])
    return slice(v1, v2)


@deprecated_func(remarks='use the independent package: `pip install argclz`', removal_version='0.5.0')
def try_int_type(arg: str) -> Union[int, str, None]:
    """for argparse (i.e., plane_index)"""
    if len(arg) == 0:
        return None
    try:
        return int(arg)
    except ValueError:
        return arg


@deprecated_func(remarks='use the independent package: `pip install argclz`', removal_version='0.5.0')
def try_float_type(arg: str) -> Union[float, str, None]:
    if len(arg) == 0:
        return None
    try:
        return float(arg)
    except ValueError:
        return arg


@overload
def literal_type(candidate: type[Literal], *, complete: bool = False):
    pass


@overload
def literal_type(*candidate: str, complete: bool = False):
    pass


@deprecated_func(remarks='use the independent package: `pip install argclz`', removal_version='0.5.0')
def literal_type(*candidate, complete: bool = False):
    if len(candidate) == 1 and not isinstance(candidate[0], str) and get_origin(candidate[0]) == Literal:
        candidate = get_args(candidate[0])

    def _literal_type(arg: str):
        if arg in candidate:
            return arg

        if not complete:
            raise ValueError

        match [it for it in candidate if it.startswith(arg)]:
            case []:
                raise ValueError()
            case [match]:
                return match
            case possible:
                raise ValueError(f'confused {possible}')

    return _literal_type
