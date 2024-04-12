import argparse
import sys
from typing import (
    Iterable, Sequence, Type, TypeVar, Union, Literal, Callable,
    overload, get_origin, get_args, Any, Optional, get_type_hints
)

__all__ = [
    'AbstractParser',
    'new_parser',
    'new_command_parser',
    'parse_args',
    'parse_command_args',
    'set_options',
    'argument', 'as_argument',
    'with_defaults',
    'print_help',
    'as_dict',
    'copy_argument',
    #
    'Argument'
]

T = TypeVar('T')
Nargs = Literal[
    '*', '+', '?', '...'
]
Actions = Literal[
    'store',
    'store_const',
    'store_true',
    'store_false',
    'append',
    'append_const',
    'extend',
    'count',
    'help',
    'version',
    #
    'boolean'
]


class AbstractParser:
    USAGE: str = None
    """parser usage."""

    DESCRIPTION: str = None
    """parser description."""

    EPILOG: str = None
    """parser epilog. Could be override as a method if its content is dynamic-generated."""

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        with_defaults(obj)
        return obj

    @classmethod
    def new_parser(cls, **kwargs) -> argparse.ArgumentParser:
        """create an ``argparse.ArgumentParser``.

        class variable: ``USAGE``, ``DESCRIPTION`` and ``EPILOG`` are used when creation.

        >>> class A(AbstractParser):
        ...     @classmethod
        ...     def new_parser(cls, **kwargs) -> argparse.ArgumentParser:
        ...         return super().new_parser(**kwargs)

        :param kwargs: keyword parameters to ArgumentParser
        :return: an ArgumentParser.
        """
        return new_parser(cls, **kwargs)

    def main(self, args: Union[list[str], tuple[list[str], list[str]]] = None, *,
             exit_on_error=True):
        """parsing the commandline input *args* and set the argument attributes,
        then call :meth:`.run()`.

        **Example**

        if overwrite with the argument default, use *args*

        >>> AbstractParser().main((['--source=allen_mouse_25um', '--region=VISal,VISam,...'], []))

        :param args: commandline arguments, or a tuple of (prepend, append) arguments
        :param exit_on_error: exit when commandline parsed fail. Otherwise, raise a ``RuntimeError``.
        """
        if args is not None:
            if isinstance(args, list):
                pass
            elif isinstance(args, tuple):
                prepend, append = args[0], args[1]
                args = [*prepend, *sys.argv[1:], *append]
            else:
                raise TypeError('')

        parser = self.new_parser(reset=True)

        try:
            result = parser.parse_args(args)
        except SystemExit as e:
            if exit_on_error:
                raise
            else:
                raise RuntimeError() from e
        else:
            set_options(self, result)
            self.post_parsing()
            self.run()

    def run(self):
        """called when all argument attributes are set"""
        pass

    def post_parsing(self):
        """called when all argument attributes are set but before :meth:`.run()`.

        It is used for a common operation for a common option class,
        for example, checking arguments before doing things.
        """
        pass


class Argument(object):
    """Descriptor (https://docs.python.org/3/glossary.html#term-descriptor).
    Carried the arguments pass to ``argparse.ArgumentParser.add_argument``.

    **Creation**

    Use :func:`~neuralib.argp.core.argument()`.

    >>> class Example:
    ...     a: str = argument('-a')

    """

    def __init__(self, *options, group: str = None, **kwargs):
        """

        :param options: options
        :param group: argument group.
        :param kwargs:
        """
        self.attr = None
        self.attr_type = Any
        self.group = group
        self.options = options
        self.kwargs = kwargs

    @property
    def default(self):
        try:
            return self.kwargs['default']
        except KeyError:
            pass

        raise ValueError

    @property
    def const(self):
        try:
            return self.kwargs['const']
        except KeyError:
            pass

        raise ValueError

    @property
    def metavar(self) -> Optional[str]:
        return self.kwargs.get('metavar', None)

    @property
    def choices(self) -> Optional[tuple[str, ...]]:
        return self.kwargs.get('choices', None)

    @property
    def required(self) -> bool:
        return self.kwargs.get('required', False)

    @property
    def help(self) -> Optional[str]:
        return self.kwargs.get('help', None)

    def __set_name__(self, owner: Type, name: str):
        self.attr = name
        self.attr_type = get_type_hints(owner).get(name, Any)

    def __get__(self, instance, owner=None):
        if instance is None:
            if owner is not None:  # ad-hoc for the document building
                self.__doc__ = self.help
            return self
        try:
            return instance.__dict__[f'__{self.attr}']
        except KeyError:
            pass

        raise AttributeError(self.attr)

    def __set__(self, instance, value):
        instance.__dict__[f'__{self.attr}'] = value

    def __delete__(self, instance):
        try:
            del instance.__dict__[f'__{self.attr}']
        except KeyError:
            pass

    def add_argument(self, ap: argparse.ArgumentParser, instance):
        """Add this into `argparse.ArgumentParser`.

        :param ap:
        :param instance:
        :return:
        """
        kwargs = self.complete_kwargs()

        try:
            return ap.add_argument(*self.options, **kwargs, dest=self.attr)
        except TypeError as e:
            if isinstance(instance, type):
                name = instance.__name__
            else:
                name = type(instance).__name__
            raise RuntimeError(f'{name}.{self.attr} : ' + repr(e)) from e

    def complete_kwargs(self) -> dict[str, Any]:
        """infer missing keywords.

        :return: kwargs
        """
        attr_type = self.attr_type
        kwargs = dict(self.kwargs)

        if 'type' not in kwargs:
            if attr_type == bool:
                if 'action' not in kwargs:
                    if kwargs.get('default', False) is False:
                        kwargs['action'] = 'store_true'
                        kwargs.setdefault('default', False)
                    else:
                        kwargs['action'] = 'store_false'
                        kwargs.setdefault('default', True)

            elif get_origin(attr_type) == Literal:
                kwargs.setdefault('choices', get_args(attr_type))

            elif get_origin(attr_type) is Union:
                type_args = get_args(attr_type)
                if len(type_args) == 2 and type_args[1] is type(None):
                    if get_origin(type_args[0]) == Literal:
                        kwargs.setdefault('choices', get_args(type_args[0]))
                    elif callable(type_args[0]):
                        kwargs['type'] = type_args[0]

            elif kwargs.get('action', None) in ['append', 'extend']:
                coll_attr_type = get_origin(attr_type)
                if coll_attr_type == list:
                    kwargs['type'] = get_args(attr_type)[0]
                else:
                    raise RuntimeError(f"cannot infer type. {self.attr} missing keyword type.")
            elif callable(attr_type):
                kwargs['type'] = attr_type

        return kwargs

    @overload
    def with_options(self,
                     option: Union[str, dict[str, str]] = None,
                     *options: str,
                     action: Actions = None,
                     nargs: Union[int, Nargs] = None,
                     const: T = None,
                     default: T = None,
                     type: Union[Type, Callable[[str], T]] = None,
                     choices: Sequence[str] = None,
                     required: bool = None,
                     help: str = None,
                     group: str = None,
                     metavar: str = None) -> 'Argument':
        pass

    def with_options(self, *options, **kwargs) -> 'Argument':
        """Modify or update keyword parameter and return a new argument.

        option flags update rule:

        1. ``()`` : do not update options
        2. ``('-a', '-b')`` : replace options
        3. ``(..., '-c')`` : append options
        4. ``({'-a': '-A'})`` : rename options
        4. ``({'-a': '-A'}, ...)`` : rename options, keep options if not in the dict.

        general form:

        ``() | (dict?, ...?, *str)``

        :param options: change option flags
        :param kwargs: change keyword parameters, use `...` to unset parameter
        :return:
        """
        kw = dict(self.kwargs)
        kw['group'] = self.group
        kw.update(kwargs)

        for k in list(kw.keys()):
            if kw[k] is ...:
                del kw[k]

        if len(self.options) > 0:
            if len(options) == 0:
                return Argument(*self.options, **kw)
            elif options[0] is ...:
                return Argument(*self.options, *options[1:], **kw)
            elif isinstance(options[0], dict):
                if len(options) == 1:
                    return Argument(*self._map_options(options[0], False), **kw)
                if len(options) == 2 and options[1] is ...:
                    return Argument(*self._map_options(options[0], True), **kw)
                if options[1] is ...:
                    return Argument(*self._map_options(options[0], True), *options[2:], **kw)
                else:
                    return Argument(*self._map_options(options[0], False), *options[1:], **kw)
            else:
                return Argument(*options, **kw)

        else:
            if len(options) > 0:
                raise RuntimeError('cannot change positional argument to optional')

            return Argument(**kw)

    def _map_options(self, mapping: dict[str, str], keep: bool) -> list[str]:
        new_opt = []
        for old_opt in self.options:
            try:
                new_opt.append(mapping[old_opt])
            except KeyError:
                if keep:
                    new_opt.append(old_opt)
        return new_opt


@overload
def argument(*options: str,
             action: Actions = None,
             nargs: Union[int, Nargs] = None,
             const: T = None,
             default: T = None,
             type: Union[Type, Callable[[str], T]] = None,
             choices: Sequence[str] = None,
             required: bool = None,
             help: str = None,
             group: str = None,
             metavar: str = None) -> T:
    pass


def argument(*options: str, **kwargs):
    """create an argument attribute.

    Example:

    >>> class Example:
    ...     # create a bool flag
    ...     bool_flag: bool = argument('-f')
    ...     # create a single value option
    ...     str_value: str = argument('-a', metavar='VALUE')
    ...     # create a single value option with type auto-casting
    ...     int_value: int = argument('-i', metavar='VALUE')
    ...     # create a position argument
    ...     pos_value: str = argument(metavar='VALUE')
    ...     # create a multiple value option
    ...     list_value: list[str] = argument('-l', metavar='VALUE', nargs=2, action='append')


    :param kwargs: Please see ``argparse.ArgumentParser.add_argument`` for detailed.
    """
    if not all([it.startswith('-') for it in options]):
        raise RuntimeError(f'options should startswith "-". {options}')
    return Argument(*options, **kwargs)


def as_argument(a) -> Argument:
    """cast argument attribute as an :class:`~neuralib.argp.core.Argument` for type checking framework/IDE."""
    if isinstance(a, Argument):
        return a
    raise TypeError


def foreach_arguments(instance: Union[T, type[T]]) -> Iterable[Argument]:
    """iterating all argument attributes in instance.

    This method will initialize Argument.

    :param instance:
    :return:
    """
    if isinstance(instance, type):
        clazz = instance
    else:
        clazz = type(instance)

    arg_set = set()
    for clz in reversed(clazz.mro()):
        if (ann := getattr(clz, '__annotations__', None)) is not None:
            for attr in ann:
                if isinstance((arg := getattr(clazz, attr, None)), Argument) and attr not in arg_set:
                    arg_set.add(attr)
                    yield arg


def new_parser(instance: Union[T, type[T]], reset=False, **kwargs) -> argparse.ArgumentParser:
    """Create ``ArgumentParser`` for instance.

    :param instance:
    :param reset: reset argument attributes. do nothing if *instance* isn't an instance.
    :param kwargs: keywords for creating :class:`argparse.ArgumentParser`.
    :return:
    """
    if isinstance(instance, AbstractParser) or (isinstance(instance, type) and issubclass(instance, AbstractParser)):
        kwargs.setdefault('usage', instance.USAGE)
        kwargs.setdefault('description', instance.DESCRIPTION)
        kwargs.setdefault('formatter_class', argparse.RawTextHelpFormatter)
        epilog = instance.EPILOG
        if callable(epilog):
            epilog = epilog()
        kwargs.setdefault('epilog', epilog)

    ap = argparse.ArgumentParser(**kwargs)

    gp = {}
    for arg in foreach_arguments(instance):
        if instance is not None and not isinstance(instance, type) and reset:
            arg.__delete__(instance)

        if arg.group is None:
            tp = ap
        elif arg.group in gp:
            tp = gp[arg.group]
        else:
            gp[arg.group] = tp = ap.add_argument_group(arg.group)

        arg.add_argument(tp, instance)

    return ap


def new_command_parser(parsers: dict[str, Union[AbstractParser, type[AbstractParser]]],
                       usage: str = None,
                       description: str = None,
                       reset=False) -> argparse.ArgumentParser:
    """Create ``ArgumentParser`` for :class:`~neuralib.argp.core.AbstractParser` s.

    :param parsers: dict of command to :class:`~neuralib.argp.core.AbstractParser`.
    :param usage: parser usage
    :param description: parser description
    :param reset: reset argument attributes. do nothing if *parsers*'s value isn't an instance.
    :return:
    """
    ap = argparse.ArgumentParser(usage=usage, description=description)
    sp = ap.add_subparsers()

    for cmd, pp in parsers.items():
        ppap = new_parser(pp, reset=reset)
        ppap.set_defaults(main=pp)
        sp.add_parser(cmd, help=pp.DESCRIPTION, parents=[ppap], add_help=False)

    return ap


def set_options(instance: T, result: argparse.Namespace) -> T:
    """set argument attributes from ``argparse.Namespace`` .

    :param instance:
    :param result:
    :return: *instance* itself.
    """
    for arg in foreach_arguments(instance):
        try:
            value = getattr(result, arg.attr)
        except AttributeError:
            pass
        else:
            arg.__set__(instance, value)

    return instance


def parse_args(instance: T, args: list[str] = None) -> T:
    """parsing the commandline input *args* and set the argument attributes.

    :param instance:
    :param args: commandline inputs
    :return:
    """
    return set_options(instance, new_parser(instance, reset=True).parse_args(args))


def parse_command_args(parsers: dict[str, Union[AbstractParser, type[AbstractParser]]],
                       args: list[str] = None,
                       usage: str = None,
                       description: str = None,
                       run_main=True) -> Optional[AbstractParser]:
    """Create ``argparse.ArgumentParser`` for :class:`~neuralib.argp.core.AbstractParser` s.
    Then parsing the commandline input *args* and setting up correspond :class:`~neuralib.argp.core.AbstractParser`.

    :param parsers: dict of command to :class:`~neuralib.argp.core.AbstractParser`.
    :param args: commandline inputs
    :param usage: parser usage
    :param description: parser description.
    :param run_main: run :meth:`~neuralib.argp.core.AbstractParser.run()`
    :return: used :class:`~neuralib.argp.core.AbstractParser`
    """
    ap = new_command_parser(parsers, usage, description, reset=True)
    res = ap.parse_args(args)

    pp: AbstractParser = getattr(res, 'main', None)
    if isinstance(pp, type):
        pp = pp()

    if pp is not None:
        set_options(pp, res)

    if run_main:
        if pp is not None:
            pp.post_parsing()
            pp.run()
        else:
            from rich import print
            print(f'[bold red]should be one of {", ".join(parsers.keys())}')

    return pp


def print_help(instance: T):
    """print help to stdout"""
    new_parser(instance).print_help(sys.stdout)


def with_defaults(instance: T) -> T:
    """Initialize all argument attributes by assign the default value if provided.

    :param instance:
    :return: *instance* itself
    """
    for arg in foreach_arguments(instance):
        ck = arg.complete_kwargs()
        try:
            value = ck['default']
        except KeyError as e:
            if 'action' in ck:
                if ck['action'] == 'store_true':
                    arg.__set__(instance, False)
                else:
                    arg.__set__(instance, True)
            else:
                arg.__set__(instance, None)
        else:
            arg.__set__(instance, value)
    return instance


def as_dict(instance: T) -> dict[str, Any]:
    """collect all argument attributes into a dictionary with attribute name to its value.

    :param instance:
    :return:
    """
    ret = {}
    for arg in foreach_arguments(instance):
        try:
            value = arg.__get__(instance)
        except AttributeError:
            pass
        else:
            ret[arg.attr] = value
    return ret


def copy_argument(opt: T, ref, **kwargs) -> T:
    """copy argument from ref to opt

    :param opt
    :param ref:
    :param kwargs:
    :return:
    """
    shadow = ShadowOption(ref, **kwargs)

    for arg in foreach_arguments(opt):
        try:
            value = getattr(shadow, arg.attr)
        except AttributeError:
            pass
        else:
            # print('set', arg.attr, value)
            arg.__set__(opt, value)
    return opt


class ShadowOption:
    """Shadow options, used to pass wrapped :class:`AbstractOptions`
    """

    def __init__(self, ref, **kwargs):
        self.__ref = ref
        self.__kwargs = kwargs

    def __getattr__(self, attr: str):
        if attr in self.__kwargs:
            return self.__kwargs[attr]

        if attr.startswith('_') and attr[1:] in self.__kwargs:
            return self.__kwargs[attr[1:]]

        if hasattr(self.__ref, attr):
            return getattr(self.__ref, attr)

        raise AttributeError(attr)
