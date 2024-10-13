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

    def _decorator(f: T) -> T:

        # class
        if isinstance(f, type):
            original_init = f.__init__

            @functools.wraps(original_init)
            def _unstable_init(self, *args: P.args, **kwargs: P.kwargs):
                warnings.warn(f"{f.__name__} is unstable and under development.", stacklevel=2)
                try:
                    original_init(self, *args, **kwargs)
                except TypeError:  # namedtuple?
                    pass

            f.__init__ = _unstable_init
            return f

        # func
        else:
            @functools.wraps(f)
            def _unstable(*args: P.args, **kwargs: P.kwargs) -> T:
                warnings.warn(f"{f.__name__} is unstable and under development.", stacklevel=2)
                return f(*args, **kwargs)

            return _unstable

    return _decorator
