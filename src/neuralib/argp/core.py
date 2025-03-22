import argparse
import collections
import sys
from types import UnionType
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

    def main(self, args: list[str] | tuple[list[str] | list[str]] | None = None, *,
             exit_on_error: bool = True):
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
        self._action_validate(parser)

        #
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

    @staticmethod
    def _action_validate(parser: argparse.ArgumentParser):
        for action in parser._actions:
            try:
                parser._get_formatter()._format_action(action)
            except ValueError as e:
                print("Error formatting help for argument:")
                print("  Option strings:", action.option_strings)
                print("  Destination:", action.dest)
                print("  Help text:", action.help)
                raise RuntimeError(repr(e))

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

    def __init__(self, *options,
                 validator: Callable[[T], bool] = None,
                 validate_on_set: bool = None,
                 group: str = None,
                 ex_group: str = None,
                 hidden: bool = False,
                 **kwargs):
        """

        :param options: options
        :param group: argument group.
        :param ex_group: mutually exclusive group.
        :param kwargs:
        """
        from .validator import Validator
        if len(options) > 0 and isinstance(options[-1], Validator):
            if validator is not None:
                raise RuntimeError()
            validator = options[-1]
            options = options[:-1]

        self.attr = None
        self.attr_type = Any
        self.group = group
        self.ex_group = ex_group
        self.validator = validator
        self.validate_on_set = validate_on_set
        self.options = options
        self.hidden = hidden
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

        if self.validate_on_set is None:
            if name.startswith('_'):
                self.validate_on_set = False
            else:
                self.validate_on_set = True

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
        if self.validate_on_set and (validator := self.validator) is not None:
            from .validator import ValidatorFailError
            try:
                fail = not validator(value)
            except ValidatorFailError:
                raise
            except BaseException as e:
                raise ValueError('validator fail') from e
            else:
                if fail:
                    raise ValueError('validator fail')

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

        if self.hidden:
            kwargs['help'] = argparse.SUPPRESS

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

        if self.ex_group is not None:
            kwargs.pop('required', False)  # has passed to the add_mutually_exclusive_group

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

            elif get_origin(attr_type) is Union or get_origin(attr_type) is UnionType:
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

        if (type_validator := self.validator) is not None:
            type_caster = kwargs.get('type', None)

            def _type_caster(value: str):
                raw_value = value
                if type_caster is not None:
                    value = type_caster(value)
                if not type_validator(value):
                    raise ValueError(f'fail validation : "{raw_value}"')
                return value

            kwargs['type'] = _type_caster

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
                     validator: Callable[[T], bool] = None,
                     validate_on_set: bool = None,
                     choices: Sequence[str] = None,
                     required: bool = None,
                     hidden: bool = None,
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
        5. ``({'-a': '-A'}, ...)`` : rename options, keep options if not in the dict.

        general form:

            ``() | (dict?, ...?, *str)``

        :param options: change option flags
        :param kwargs: change keyword parameters, use `...` to unset parameter
        :return:
        """
        kw = dict(self.kwargs)
        kw['group'] = self.group
        kw['ex_group'] = self.ex_group
        kw['validator'] = self.validator
        kw['validate_on_set'] = self.validate_on_set
        kw['hidden'] = self.hidden
        kw.update(kwargs)

        for k in list(kw.keys()):
            if kw[k] is ...:
                del kw[k]

        cls = type(self)

        if len(self.options) > 0:
            match options:
                case ():
                    return cls(*self.options, **kw)
                case (e, *o) if e is ...:
                    return cls(*self.options, *o, **kw)
                case (dict(d), ):
                    return cls(*self._map_options(d, False), **kw)
                case (dict(d), e) if e is ...:
                    return cls(*self._map_options(d, True), **kw)
                case (dict(d), e, *o) if e is ...:
                    return cls(*self._map_options(d, True), *o, **kw)
                case (dict(d), *o):
                    return cls(*self._map_options(d, False), *o, **kw)
                case _:
                    return cls(*options, **kw)
        else:
            if len(options) > 0:
                raise RuntimeError('cannot change positional argument to optional')

            return cls(**kw)

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
             action: Actions = ...,
             nargs: Union[int, Nargs] = ...,
             const: T = ...,
             default: T = ...,
             type: Union[Type, Callable[[str], T]] = ...,
             validator: Callable[[T], bool] = ...,
             validate_on_set: bool = True,
             choices: Sequence[str] = ...,
             required: bool = False,
             hidden: bool = False,
             help: str = ...,
             group: str = None,
             ex_group: str = None,
             metavar: str = ...) -> T:
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
    if not all([it.startswith('-') for it in options if isinstance(it, str)]):
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

    groups: dict[str, list[Argument]] = collections.defaultdict(list)

    # setup non-grouped arguments
    mu_ex_groups: dict[str, argparse._ActionsContainer] = {}
    for arg in foreach_arguments(instance):
        if instance is not None and not isinstance(instance, type) and reset:
            arg.__delete__(instance)

        if arg.group is not None:
            groups[arg.group].append(arg)
            continue
        elif arg.ex_group is not None:
            try:
                tp = mu_ex_groups[arg.ex_group]
            except KeyError:
                # XXX current Python does not support add title and description into mutually exclusive group
                #   so the message in ex_group is dropped.
                mu_ex_groups[arg.ex_group] = tp = ap.add_mutually_exclusive_group()

            if arg.required:
                tp.required = True
        else:
            tp = ap

        arg.add_argument(tp, instance)

    # setup grouped arguments
    for group, args in groups.items():
        pp = ap.add_argument_group(group)
        mu_ex_groups: dict[str, argparse._ActionsContainer] = {}

        for arg in args:
            if arg.ex_group is not None:
                try:
                    tp = mu_ex_groups[arg.ex_group]
                except KeyError:
                    mu_ex_groups[arg.ex_group] = tp = pp.add_mutually_exclusive_group()

                if arg.required:
                    tp.required = True
            else:
                tp = pp

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
        except KeyError:
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
