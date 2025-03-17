import functools
import inspect
import warnings

__all__ = ['deprecated_class',
           'deprecated_func',
           'deprecated_aliases']


def _build_deprecated_message(target: str,
                              alternation: str | None = None,
                              remarks: str | None = None,
                              removal: str | None = None) -> str:
    msg = f'{target} is deprecated'

    if removal is not None:
        msg += f' and will be removed in a future release (after version {removal}).'
    else:
        msg += '.'

    if alternation is not None:
        msg += f' Please use "{alternation}" instead.'

    if remarks is not None:
        msg += f' NOTE: {remarks}.'

    return msg


def deprecated_class(*,
                     new: str | None = None,
                     remarks: str | None = None,
                     removal_version: str | None = None):
    """Mark deprecated class

    :param new: The renamed new usage
    :param remarks: Further remarks to be shown
    :param removal_version: Optional version or date when the function is planned to be removed
    """

    def _decorator(cls):
        ori_init = cls.__init__

        @functools.wraps(ori_init)
        def new_init(self, *args, **kwargs):
            msg = _build_deprecated_message(cls.__qualname__, new, remarks, removal_version)

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
                    new: str | None = None,
                    remarks: str | None = None,
                    removal_version: str = None):
    """Mark deprecated functions.

    :param new: The renamed new usage
    :param remarks: The reason why the function is deprecated
    :param removal_version: Optional version or date when the function is planned to be removed
    """

    def _decorator(f):

        @functools.wraps(f)
        def _deprecated_func(*args, **kwargs):
            msg = _build_deprecated_message(f.__qualname__, new, remarks, removal_version)

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


def deprecated_aliases(**aliases: str):
    """
    Mark deprecated argument names and map them to new argument names in a function

    This decorator allows you to support old argument names while transitioning to new ones.
    It will raise a ``DeprecationWarning`` if the old argument name is used and will automatically
    map the value to the new argument name.

    :param aliases: mapping of `old argument names` to `new argument names`.
    :return: Decorated function with support for deprecated argument names.

    :raises RuntimeError: If a new argument name does not exist in the function's signature.
    :raises ValueError: If both old and new argument names are provided simultaneously.
    """

    def _decorator(f):
        sig = inspect.signature(f)

        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            for old_arg, new_arg in aliases.items():

                if new_arg not in sig.parameters:
                    raise RuntimeError(f'New argument: {new_arg} is not in the function arg.')

                if new_arg in kwargs and old_arg in kwargs:
                    raise ValueError(f'Cannot specify both {old_arg} and {new_arg} at the same time')

                if old_arg in kwargs:
                    warnings.warn(
                        f'"{old_arg}" is deprecated and will be removed in future version. Use "{new_arg}" instead',
                        DeprecationWarning,
                        stacklevel=2
                    )

                    # If the new argument is not already in kwargs, move the value from the old argument
                    if new_arg not in kwargs:
                        kwargs[new_arg] = kwargs.pop(old_arg)

            return f(*args, **kwargs)

        return _wrapper

    return _decorator
