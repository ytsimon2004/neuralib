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


def unstable(doc: bool = True, runtime: bool = True, mark_all: bool = True) -> Callable[[T], T]:
    """
    A decorator that mark class/function unstable.

    :param doc: Add unstable message in function/class document.
    :param runtime: Add runtime warning when invoking function/initialization.
    :param mark_all: Mark all public methods when decorate on a class.
    """

    def _decorator(obj: T) -> T:

        if doc:
            new_doc = 'UNSTABLE.'
            if obj.__doc__ is not None:
                new_doc += f'\n{obj.__doc__}'

            obj.__doc__ = new_doc

        # class
        if isinstance(obj, type):
            if mark_all:
                for meth in dir(obj):
                    if callable(f := getattr(obj, meth, None)) and not meth.startswith('_'):
                        setattr(obj, meth, unstable()(f))

            if runtime:
                original_init = obj.__init__

                @functools.wraps(original_init)
                def _unstable_init(self, *args: P.args, **kwargs: P.kwargs) -> None:
                    warnings.warn(f"{obj.__qualname__} is unstable and under development.", stacklevel=2)
                    original_init(self, *args, **kwargs)

                obj.__init__ = _unstable_init

            return obj

        # func/meth
        else:
            if runtime:
                @functools.wraps(obj)
                def _unstable_meth(*args: P.args, **kwargs: P.kwargs) -> T:
                    warnings.warn(f"{obj.__qualname__} is unstable and under development.", stacklevel=2)
                    return obj(*args, **kwargs)

                return _unstable_meth
            else:
                return obj

    return _decorator
