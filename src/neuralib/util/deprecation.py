from __future__ import annotations

import functools
import warnings

__all__ = ['deprecated_class',
           'deprecated_func']


def deprecated_class(*,
                     new_class: str | None = None,
                     remarks: str | None = None,
                     removal_version: str | None = None):
    """A decorator to mark a class as deprecated and suggest using a new class

    :param new_class: The renamed new class
    :param remarks: Further remarks to be shown
    :param removal_version: Optional version or date when the function is planned to be removed
    """

    def _decorator(cls):
        ori_init = cls.__init__

        @functools.wraps(ori_init)
        def new_init(self, *args, **kwargs):
            msg = f'{cls.__qualname__} is deprecated and will be removed in a future release'

            if removal_version is not None:
                msg += f': {removal_version}.'

            if new_class is not None:
                msg += f' Please use {new_class} instead.'

            if remarks is not None:
                msg += f' NOTE: {remarks}.'

            warnings.warn(
                msg,
                DeprecationWarning,
                stacklevel=2
            )

            ori_init(self, *args, **kwargs)

        cls.__init__ = new_init

        #
        if cls.__doc__ is None:
            cls.__doc__ = 'DEPRECATED.'
        else:
            cls.__doc__ = 'DEPRECATED.' + cls.__doc__

        return cls

    return _decorator


def deprecated_func(*,
                    new_function: str | None = None,
                    remarks: str | None = None,
                    removal_version: str = None):
    """Mark deprecated functions.

    :param new_function: The renamed new function
    :param remarks: The reason why the function is deprecated
    :param removal_version: Optional version or date when the function is planned to be removed
    """
    if remarks is None:
        remarks = 'This function is deprecated and may be removed in future versions.'

    if removal_version is not None:
        remarks += f' Scheduled for removal in version {removal_version}.'

    def _decorator(f):

        @functools.wraps(f)
        def _deprecated_func(*args, **kwargs):

            msg = f'{f.__qualname__} is deprecated and will be removed in a future release'

            if removal_version is not None:
                msg += f': {removal_version}.'

            if new_function is not None:
                msg += f' Please use {new_function} instead.'

            if remarks is not None:
                msg += f' NOTE: {remarks}.'

            warnings.warn(
                msg,
                DeprecationWarning,
                stacklevel=2
            )

            return f(*args, **kwargs)

        if f.__doc__ is None:
            _deprecated_func.__doc__ = "DEPRECATED."
        else:
            _deprecated_func.__doc__ = "DEPRECATED. " + f.__doc__

        return _deprecated_func

    return _decorator
