from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, TypeVar, Callable

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import ParamSpec
    else:
        from typing_extensions import ParamSpec

    P = ParamSpec("P")
    T = TypeVar("T")

__all__ = ['unstable']


def unstable() -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Mask class/function unstable"""

    def _decorator(obj: Callable[P, T] | type) -> Callable[P, T] | type:

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
