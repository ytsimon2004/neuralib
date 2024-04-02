"""
Value dispatch function
=======================

:author:
    Ta-Shun Su

Work with AbstractParser.

>>> from neuralib.argp import AbstractParser
>>> class Test(AbstractParser, DispatchOption):
...     target: str = DispatchOption.argument(
...         '--run'
...     )
...
...     @classmethod
...     def EPILOG(cls): # build parser epilog
...         return f'''\
... Command (--run)
... {textwrap.indent(Test.parser_command_epilog(), "  ")}
... '''
...
...     def run(self):
...         # dispaatch function call according to self.target
...         self.invoke_command(self.target)
...
...     @DispatchOption.dispatch('A')
...     def run_a(self):
...         '''doc A.'''
...         print('a')
...     @DispatchOption.dispatch('B')
...     def run_b(self):
...         '''doc B'''
...         print('b')
"""

from typing import Union, NamedTuple, Optional, Callable, TypeVar, Any, Final, Type

from .core import Argument

__all__ = [
    'dispatch',
    'list_commands',
    'find_command',
    'invoke_command',
    'invoke_group_command',
    'DispatchOption',
    'DispatchCommandNotFound'
]

T = TypeVar('T')
missing = object()

NEURALIB_DISPATCH_INFO = '__NEURALIB_DISPATCH_INFO__'


def dispatch(command: str,
             alias: Union[str, list[str]] = None,
             group: str = None):
    """A decorator that mark a function a dispatch target function.

    All functions decorated in same dispatch group should have save
    function signature (at least for non-default parameters). For example:

    >>> @dispatch('A')
    ... def function_a(self, a, b, c=None):
    ...     pass
    ... @dispatch('B')
    ... def function_b(self, a, b, d=None):
    ...     pass
    ... def run_function(self):
    ...     invoke_command(self, 'A', a, b)


    """
    if len(command) == 0:
        raise ValueError('empty command string')

    if alias is None:
        alias = []
    elif isinstance(alias, str):
        alias = [alias]

    def _dispatch(f):
        if hasattr(f, NEURALIB_DISPATCH_INFO):
            raise RuntimeError()

        setattr(f, NEURALIB_DISPATCH_INFO, DispatchCommand(command, tuple(alias), group, f))
        return f

    return _dispatch


class DispatchCommand(NamedTuple):
    command: str
    aliases: tuple[str, ...]
    group: Optional[str]
    func: Callable

    def __call__(self, host: T, *args, **kwargs) -> Any:
        return self.func(host, *args, **kwargs)

    @property
    def doc(self) -> Optional[str]:
        return self.func.__doc__


class DispatchCommandNotFound(RuntimeError):
    def __init__(self, command: str, group: Optional[str] = None):
        if group is None:
            message = f'command {command} not found'
        else:
            message = f'command {group}:{command} not found'

        super().__init__(message)


def list_commands(host: Union[T, Type[T]], group: Optional[str] = missing) -> list[DispatchCommand]:
    """list all dispatch-decoratored function info in *host*.

    :param host:
    :param group: dispatch group.
    :return: list of DispatchCommand
    """
    ret = []

    info: DispatchCommand

    if isinstance(host, type):
        host_type = host
    else:
        host_type = type(host)

    for attr in dir(host_type):
        attr_value = getattr(host_type, attr)
        if (info := getattr(attr_value, NEURALIB_DISPATCH_INFO, None)) is not None:
            if group is missing or group == info.group:
                ret.append(info)

    return ret


def find_command(host: T, command: str, group: Optional[str] = missing) -> Optional[DispatchCommand]:
    """find dispatch-decoratored function in *host* according to *command*.

    :param host:
    :param command: command or command alias
    :param group: dispatch group
    :return: found DispatchCommand
    """
    info: DispatchCommand

    host_type = type(host)
    for attr in dir(host_type):
        attr_value = getattr(host_type, attr)
        if (info := getattr(attr_value, NEURALIB_DISPATCH_INFO, None)) is not None:
            if group is missing or group == info.group:
                if command == info.command or command in info.aliases:
                    return info

    return None


def invoke_command(host: T, command: str, *args, **kwargs) -> Any:
    """invoke a dispatch-decoratored function in default group.

    :param host:
    :param command: command or command alias
    :param args: dispatch-decoratored function positional arguments
    :param kwargs: dispatch-decoratored function keyword arguments
    :return: function return
    :raise DispatchCommandNotFound:
    """
    if (info := find_command(host, command, None)) is None:
        raise DispatchCommandNotFound(command)
    return info(host, *args, **kwargs)


def invoke_group_command(host: T, group: str, command: str, *args, **kwargs) -> Any:
    """invoke a dispatch-decoratored function in certain group.

    :param host:
    :param group: dispatch group
    :param command: command or command alias
    :param args: dispatch-decoratored function positional arguments
    :param kwargs: dispatch-decoratored function keyword arguments
    :return: function return
    :raise DispatchCommandNotFound:
    """
    if (info := find_command(host, command, group)) is None:
        raise DispatchCommandNotFound(command, group)
    return info(host, *args, **kwargs)


class DispatchArgument(Argument):
    pass


class DispatchOption:
    argument: Final = DispatchArgument

    def list_commands(self, group: Optional[str] = missing) -> list[DispatchCommand]:
        """list all dispatch-decoratored function info in *host*.

        :param group: dispatch group.
        :return: list of DispatchCommand
        """
        return list_commands(self, group)

    def find_command(self, command: str, group: Optional[str] = missing) -> Optional[DispatchCommand]:
        """find dispatch-decoratored function in *host* according to *command*.

        :param command: command or command alias
        :param group: dispatch group
        :return: found DispatchCommand
        """
        return find_command(self, command, group)

    def invoke_command(self, command: str, *args, **kwargs) -> Any:
        """invoke a dispatch-decoratored function in default group.

        :param command: command or command alias
        :param args: dispatch-decoratored function positional arguments
        :param kwargs: dispatch-decoratored function keyword arguments
        :return: function return
        :raise DispatchCommandNotFound:
        """
        return invoke_command(self, command, *args, **kwargs)

    def invoke_group_command(self, group: str, command: str, *args, **kwargs) -> Any:
        """invoke a dispatch-decoratored function in certain group.

        :param group: dispatch group
        :param command: command or command alias
        :param args: dispatch-decoratored function positional arguments
        :param kwargs: dispatch-decoratored function keyword arguments
        :return: function return
        :raise DispatchCommandNotFound:
        """
        return invoke_group_command(self, group, command, *args, **kwargs)

    @classmethod
    def dispatch(cls, command: str,
                 alias: Union[str, list[str]] = None,
                 group: str = None):
        return dispatch(command, alias, group)

    @classmethod
    def parser_command_epilog(cls, group: Optional[str] = missing) -> str:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        console = Console()

        with console.capture() as capture:
            table = Table(show_header=False, show_edge=False, box=box.SIMPLE)
            table.add_column('command', min_width=18)
            table.add_column('desp')

            for info in list_commands(cls, group):
                header = info.command
                if len(info.aliases) > 0:
                    header += ' (' + ', '.join(info.aliases) + ')'

                content = info.doc or ''
                content = content.split('\n')[0].strip()

                table.add_row(header, content)

            console.print(table)

        return capture.get()
