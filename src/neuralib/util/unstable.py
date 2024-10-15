from __future__ import annotations

import functools
import sys
import warnings
from typing import TypeVar, Callable, Union, Type

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

__all__ = ['unstable']

P = ParamSpec("P")
R = TypeVar("R")
F = TypeVar('F', bound=Callable[P, R])
T = TypeVar('T', bound=Union[Type[R], F])


def unstable(doc=True, runtime=True, mark_all=True) -> Callable[[T], T]:
    """
    A decorator that mark class/function unstable.

    :param doc: add unstable message in function/class document.
    :param runtime: add runtime warning when invoking function/initialization.
    :param mark_all: mark all public methods when decorate on a class.
    """

    def _decorator(obj: T) -> T:

        doc = 'UNSTABLE.'
        if obj.__doc__ is not None:
            doc += f'\n{obj.__doc__}'

        obj.__doc__ = doc

        # class
        if isinstance(obj, type):
            for meth in dir(obj):
                if callable(f := getattr(obj, meth, None)) and not meth.startswith('_'):
                    setattr(obj, meth, unstable()(f))

            return obj

        # func
        else:
            @functools.wraps(obj)
            def _unstable_meth(*args: P.args, **kwargs: P.kwargs) -> T:
                warnings.warn(f"{obj.__qualname__} is unstable and under development.", stacklevel=2)
                return obj(*args, **kwargs)

            return _unstable_meth

    return _decorator
