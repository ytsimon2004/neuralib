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

    def _decorator(f: Callable[P, T] | type) -> Callable[P, T] | type:

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
