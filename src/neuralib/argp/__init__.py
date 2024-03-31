"""
Annotation-based argparse
=========================

:author:
    Ta-Shun Su

This module provide a way to integrate python `argparse` module into class attribute
and annotation type, that allow options have type information, and allow parser combination
easily.

class Argument and function argument
------------------------------------

A simple option class that contains several options carried by their attributes.

>>> class ExampleOptions:
...     ANIMAL: str = argument('--ANIMAL')
...     EXP_DATE: str = argument('--EXP_DATE')
...     OUTPUT_DIR: str = argument()

In `ExampleOptions`, `ANIMAL` is an attribute with type annotation `str`. it has
a class variable :class:`Argument` (:func:`argument`'s return) which contains the
arguments of :meth:`argparse.ArgumentParser.add_argument`. For now, we have an optional
argument `--ANIMAL` which accept one argument. `EXP_DATE` is another optional argument
`--EXP_DATE`. `OUTPUT_DIR` is a potional argument because it doesn't have dashed options.

>>> opt = parse_args(ExampleOptions())
... print(opt.animal)

After class declared, you can use :func:`parse_args` to parse cli arguments. This
function will create an :class:`argparse.ArgumentParser` and find out all argument
attributes. Then set the attributes from parsed result.

Commandline usage
~~~~~~~~~~~~~~~~~

In bash, you can call this option class ::

    python -m module.path --ANIMAL name --EXP_DATE date output

:func:`print_help` does the similar things but print the help document to the stdout.

>>> print_help(opt)

Or use `-h` options ::

    python -m module.path -h

Annotation type infering
~~~~~~~~~~~~~~~~~~~~~~~~

In general, you can think :func:`argument` just a delegate funcion that passes the arguments
to the :meth:`argparse.ArgumentParser.add_argument`. However, this function will try to
infer missing arguments based on the annotation type when creating the :class:`argparse.ArgumentParser`.
For now, this module is not powerful to handle all possible case. there are support type
(Please see :meth:`Argument.complete_kwargs` for detailed):

1. `bool`: infer parameter `action` to `store_true`, `default` to `False`.
2. `Literal[...]` : infer parameter `choices`.
3. `Optional[T]`: infer parameter `type` to `T` if it is callable. If it is `Literal`, apply 2.
4. `callable(T)` : infer parameter `type` to `T`
5. parameter `dest` always use attribute name.

Additionally, :class:`Argument` also provide a parameter `group` to reduce the complexity of
create subgrouping parser.

Option class compose
~~~~~~~~~~~~~~~~~~~~

Option class can be composed by inherition. Child option class can also change the value from parent's
argument. As well as disable it (by replacing a value)

>>> class MoreOptions(ExampleOptions):
...     # additional optional option
...     verbose: bool = argument('-v', '--verbose')
...     # change default value
...     ANIMAL: str = as_argument(ExampleOptions.animal).with_options(default='YW00')
...     # and disable an option
...     OUTPUT_DIR: str = 'output' # just replace with a value

Change options name is more complicate, because you might want to add more name, remove some name,
or rename some name. :meth:`Argument.with_options` allow you to do that:

>>> class ChangeExample(ExampleOptions):
...     # replace option name: --animal
...     ANIMAL: str = as_argument(ExampleOptions.animal).with_options('--animal')
...     # add more option name: -A, --ANIMAL
...     ANIMAL: str = as_argument(ExampleOptions.animal).with_options(..., '-A')
...     # rename option: --animal
...     ANIMAL: str = as_argument(ExampleOptions.animal).with_options({
...         '--ANIMAL': '--animal'
...     })

Utility function for option class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`ExampleOptions` doesn't declated neither `__str__` nor `__repr__`, so it is not convenient to debug.
funciont :func:`as_dict` provide a way to take argument attribute's value into a dictionary.

>>> as_dict(opt)
{'ANIMAL': ..., 'EXP_DATE': ..., 'OUTPUT_DIR': ...}

Option class is not restricted into only one use case. It works like a normal class.

class AbstractParser
--------------------

This class provide a main like class that has more control on :class:`argparse.ArgumentParser`
creation.

>>> class ExampleParser(AbstractParser, ExampleOptions):
...     DESCRIPTION = 'Example parser'
...     def run(self):
...         ...
>>> if __name__ == '__main__':
...     ExampleParser().main()

Subcommands
-----------

This module isn't fully support sub-command feature, but only provide a simple way for specific case:

>>> parse_command_args(
...     description='top level parser',
...     parsers=dict(ep=ExampleParser)
... )

In bash::

    python -m module.path ep --ANIMAL name --EXP_DATE date output

"""
from ._type import *
from .core import *
