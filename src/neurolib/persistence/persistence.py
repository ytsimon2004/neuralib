"""
Persistence Class
=================

Define a persistence class
-----------------------

Import

>>> import numpy as np
>>> from neurolib import persistence

Define class

>>> @persistence.persistence_class
... class Example:
...     # key. use filename keyword to display key in filename.
...     use_animal: str = persistence.field(validator=True, filename=True)
...     use_session: str = persistence.field(validator=True)
...     use_date: str = persistence.field(validator=True, filename=True)
...     # data
...     channels: list[int]
...     data: np.ndarray

Load/Save

>>> example = Example(use_animal='TS00', use_session='', use_date='1234')
>>> save(example, 'example.pkl')
>>> example_2 = load(Example, 'example.pkl') # example and example_2 should be content identical


Cooperate with PersistenceOptions
---------------------------------

>>> from rscvp.util.cli import CommonOptions
>>> from rscvp.util.cli_presistence import PersistenceOptions
>>> class ExampleHandle(CommonOptions, PersistenceOptions[Example]):
...     def empty_cache(self) -> Example:
...         return Example(use_animal='TS00', use_session='', use_date='1234') # with attribute initialization
...     def _compute_cache(self, result: Example) -> Example:
...         result.channels = [0, 1, 2]
...         result.riglog = np.array(result.channels)
...         return result

Dynamic generated methods for persistence class
---------------------------------------------

1. `__init__({foreach persistence.field})`

    >>> class Example:
    ...     ... # as same as above
    ...     def __init__(self, use_animal:str,  use_session: str, use_date: str): # auto generated
    ...         ...

2. `__str__` return filename

    >>> class Example:
    ...     ... # as same as above
    ...     def __str__(self): # auto generated
    ...         return filename(self)

3. `__repr__`

    >>> class Example:
    ...     ... # as same as above
    ...     def __repr__(self): # auto generated
    ...         return 'Example{' + f'use_animal={self.use_animal}, use_session={self.use_session}, use_date={self.use_date}' + "}"

4. `_replace` when persistence define empty `_replace` methods. It is NamedTuple._replace like function.

    >>> class Example:
    ...     ... # as same as above
    ...     def _replace(self, **kwargs): pass # empty method
    ...     def _replace(self, *,  # replaced by generated
    ...                  use_animal=missing,
    ...                  use_session=missing,
    ...                  use_date=missing,
    ...                  channels=missing,
    ...                  data=missing) -> Example:
    ...         ...

Auto increment field
--------------------

For some reason you want to save a persistence result that came from same data source but different
contents, which are usualy generated by random or suffle process. To save them all separately, you may
need a field value that keep track that it is n-th persistence result. :func:`autoinc_field` is proposed
to help this case.


>>> @persistence_class
... class Result:
...     a: str = field(validator=True, filename=T)
...     b: int = autoinc_field()
...     c: str
...     def __init__(self, a: str): # auto generated signature
...     def _replace(self, *, a: str, c:str): # auto generated signature

There are some rule when a persistence class has an autoinc_field.

1. only one autoinc_field in a persistence class allowed.
2. field type only int is allowed.
3. raise error when load a result without autoinc field resolved.
4. autoinc field is auto resolved when saving. Its value is max(found) + 1

Pickle format
-------------

Persistence class is transformed into a dict by `as_dict`, which as a root instance to be saved into
a pickle file.

**IMPORTANT**

If a persistance class has a custom `__init__` function which signature is differed from auto generated,
you need to define a classmethod `from_dict` for creating that persistance class.

>>> @persistence.persistence_class
... class Example:
...     a: int = persistence.field(validator=True, filename=True)
...     b: int = persistence.field(validator=True, filename=True)
...     c: int
...     def __init__(self): ... # custom __init__
...     @classmethod
...     def from_dict(cls, data:dict[str, Any]) -> 'Example':
...         # data = {'a', 'b', 'c'}


"""
from __future__ import annotations

import abc
import inspect
from pathlib import Path
from typing import Type, TypeVar, Union, Callable, Optional, Generic, Any, Iterable, get_type_hints

import numpy as np

from neurolib.util.func import create_fn
from neurolib.util.util_type import is_iterable
from neurolib.util.util_verbose import fprint

__all__ = [
    'field',
    'autoinc_field',
    'persistence_class',
    'ensure_persistence_class',
    'as_dict',
    'from_dict',
    'load',
    'save',
    'filename',
    'AutoIncFieldNotResolvedError',
    'auto_generated_content',
    'PersistenceHandler',
    'PickleHandler',
    'GzipHandler',
    #
    'ETLConcatable',
    'validate_concat_etl_persistence',
]

T = TypeVar('T')
P = Union[str, Path]

missing = inspect.Parameter.empty

VALIDATOR = Callable[[T, T], bool]


def field(validator: Union[bool, VALIDATOR] = False,
          filename_prefix: str = '',
          filename: Union[bool, Callable[..., str]] = False) -> T:
    """Cache class's exported field. Used as keys to find correspond persistence.

    :param validator: validate this field. use __eq__ by default. Can be a callable as a customized validator.
    :param filename_prefix: prefix word of *filename*
    :param filename: display this value on filename. Can be a callable that return the string.
    :return:
    """
    return PersistentField(validator, filename_prefix, filename, init=True)


def autoinc_field(filename_prefix: str = '') -> int:
    """make a field as auto increment, which can be used to save and distinguish between same source data but
    different random/shuffle persistence result.

    :param filename_prefix:
    :return:
    """
    # noinspection PyTypeChecker
    return PersistentField(validator=True, filename_prefix=filename_prefix, filename=True, init=False, autoinc=True)


class PersistentField(Generic[T]):
    """exported field of persistence class."""

    __slots__ = ('field_name', 'field_type', 'validator', 'filename_prefix', 'filename', 'init', 'autoinc')

    def __init__(self,
                 validator: Union[bool, VALIDATOR] = False,
                 filename_prefix: str = '',
                 filename: Union[bool, Callable[..., str]] = False,
                 init=True,
                 autoinc=False):
        """

        :param validator: validate this field. use __eq__ by default. Can be a callable as a customized validator.
        :param filename_prefix: prefix word of *filename*
        :param filename: display this value on filename. Can be a callable that return the string.
        :param init: put this field into class __init__
        """
        self.validator = validator
        self.filename_prefix = filename_prefix
        self.filename = filename
        self.init = init
        self.autoinc = autoinc

    def __set_name__(self, owner: Type, name: str):
        self.field_name = name
        if self.autoinc:
            if (field_type := get_type_hints(owner).get(name, Any)) != int:
                raise RuntimeError(f'type of autoinc field {name} should be int, but {field_type}')
        else:
            self.field_type = owner.__annotations__.get(name, Any)

    def validate(self, v: T, u: T) -> bool:
        """Validate this two value.

        :param v:
        :param u:
        :return:
        """
        if self.validator is False:
            return True
        elif self.validator is True:
            return v == u
        elif callable(self.validator):
            return self.validator(v, u)
        else:
            raise RuntimeError()


class AutoIncFieldNotResolvedError(RuntimeError):
    def __init__(self, instance, field: Union[str, PersistentField], message: str = None):
        if isinstance(field, PersistentField):
            field = field.field_name

        if message is None:
            message = f'{type(instance).__name__} autoinc field {field} is not resolved'

        super().__init__(message)

        self.instance = instance
        self.field = field


def persistence_class(cls: Type = None, /, *,
                      name: str = None,
                      filename_field_splitter='-'):
    """A class decorator.

    Decorated class ...

    :param cls: Persistence class.
    :param name: class name as filename.
    :param filename_field_splitter: the field splitter on filename.
    :return:
    """

    def decorator(cls: Type):
        cls._ast_persistence_cls_info_ = pc = PersistentClass(cls, name, filename_field_splitter)

        prev_autoinc_field = None
        for attr_name, attr_type in cls.__annotations__.items():
            if isinstance((attr_value := getattr(cls, attr_name, None)), PersistentField):
                delattr(cls, attr_name)

                f = attr_value
            else:
                f = PersistentField(init=False)
                f.field_name = attr_name
                f.field_type = attr_type

            if f.autoinc:
                if prev_autoinc_field is None:
                    prev_autoinc_field = f
                else:
                    raise RuntimeError('duplicated auto_inc_field')

            pc.fields.append(f)

        if cls.__init__ == object.__init__:
            cls.__init__ = _persistence_class_init(pc)
        if cls.__str__ == object.__str__:
            cls.__str__ = _persistence_class_str(pc)
        if cls.__repr__ == object.__repr__:
            cls.__repr__ = _persistence_class_repr(pc)
        if hasattr(cls, '_replace'):
            cls._replace = _persistence_class_replace(pc)
        return cls

    if cls is None:
        return decorator
    else:
        return decorator(cls)


class PersistentClass(Generic[T]):
    """persistence info class"""

    __slots__ = ('persistence_cls', 'cls_name', 'filename_field_splitter', 'fields')

    def __init__(self,
                 persistence_cls: Type,
                 persistence_name: str = None,
                 filename_field_splitter='-'):
        self.persistence_cls = persistence_cls
        self.cls_name = persistence_name if persistence_name is not None else persistence_cls.__name__
        self.filename_field_splitter = filename_field_splitter
        self.fields: list[PersistentField] = []

    def fields_name(self) -> list[str]:
        return [it.field_name for it in self.fields]

    def get_field(self, name: str) -> Optional[PersistentField]:
        for f in self.fields:
            if f.field_name == name:
                return f
        return None

    def autoinc_field(self) -> Optional[PersistentField]:
        for f in self.fields:
            if f.autoinc:
                return f
        return None

    def is_autoinc_field_resolved(self, result: Optional[T], **kwargs) -> bool:
        if (af := self.autoinc_field()) is None:
            return True
        return getattr(result, af.field_name, missing) is not missing or af.field_name in kwargs

    def validate(self, v: T, u: Union[T, dict[str, Any]]) -> bool:
        """validate that does data u is as same as data v.

        :param v: reference data
        :param u: tested data
        :return: False if validation fail.
        """
        for f in self.fields:
            fv = getattr(v, f.field_name, None)

            if isinstance(u, dict):
                fu = u.get(f.field_name, None)
            else:
                fu = getattr(u, f.field_name, None)

            if not f.validate(fv, fu):
                fprint(f'{self.cls_name}.{f.field_name} validate fail: {fv} != {fu}')
                return False

        return True

    def filename(self, data: Optional[T], **kwargs) -> str:
        """build filename for persistence instance.

        :param data: persistence instance.
        :param kwargs: overwrote keywords.
        :return: filename.
        :raise RuntimeError: filename field missing.
        :raise TypeError: wrong field.filename type.
        """
        # legacy name
        ret = ['cache', self.cls_name]

        for f in self.fields:
            if f.filename:
                if (field_value := kwargs.get(f.field_name, missing)) is missing:
                    if data is not None:
                        field_value = getattr(data, f.field_name, None)
                    else:
                        raise RuntimeError(f'field {f.field_name} is required for filename')

                if field_value is missing:
                    if f.autoinc:
                        s = '{}'
                    else:
                        s = '*'
                elif f.filename is True:
                    s = str(field_value)
                elif callable(f.filename):
                    s = str(f.filename(field_value))
                else:
                    raise TypeError()

                ret.append(f.filename_prefix + s)

        return self.filename_field_splitter.join(ret)


def _persistence_class_init(pc: PersistentClass):
    init_fields = ['self'] + [f.field_name for f in pc.fields if f.init]
    code = []
    for name in init_fields[1:]:
        code.append(f'self.{name} = {name}')
    if (af := pc.autoinc_field()) is not None:
        code.append(f'self.{af.field_name} = missing')
    return create_fn('__init__', init_fields, '\n'.join(code),
                     locals=dict(missing=missing))


def _persistence_class_str(pc: PersistentClass):
    return create_fn('__str__', (['self'], str), 'return filename(self)', locals={'filename': filename})


def _persistence_class_repr(pc: PersistentClass):
    init_fields = [f.field_name for f in pc.fields if f.init]

    code = [f'return ("{pc.cls_name}' + '{"']
    for i, name in enumerate(init_fields):
        comma = '", " ' if i > 0 else ''
        code.append(f'{comma}f"{name}={{self.{name}}}"')
    code.append('"}")')

    return create_fn('__repr__', (['self'], str), ' '.join(code))


def _persistence_class_replace(pc: PersistentClass):
    init_fields = [f.field_name for f in pc.fields if f.init]
    data_fields = [f.field_name for f in pc.fields if not f.init and not f.autoinc]
    code = [f'ret = {pc.cls_name}(']
    for name in init_fields:
        code.append(f'{name} if {name} is not missing else self.{name},')
    code.append(')')
    for name in data_fields:
        code.append(f'ret.{name} = {name} if {name} is not missing else self.{name}')
    code.append('return ret')

    return create_fn('_replace',
                     (['self', '*'] + [(name, None, 'missing') for name in (init_fields + data_fields)], pc.cls_name),
                     '\n'.join(code),
                     locals={pc.cls_name: pc.persistence_cls, 'missing': missing})


def auto_generated_content():
    """It is used to mark the function which its function body is auto generated.

    :return: nothing
    """
    raise RuntimeError('It is auto generated content')


def ensure_persistence_class(data: Union[T, type[T]]) -> PersistentClass[T]:
    """ensure **data** is a persistence class.

    :param data: instance or type
    :return: persistence info
    :raise RuntimeError: not a persistence class
    """
    if not isinstance(data, type):
        data = type(data)

    try:
        cls_info: PersistentClass = data._ast_persistence_cls_info_
    except AttributeError as e:
        raise RuntimeError(f'not a persistence_class : {data.__name__}') from e

    return cls_info


def as_dict(data: T) -> dict[str, Any]:
    """transform persistence *data* into dictionary, which field as key.

    :param data: persistence instance
    :return: dict
    """
    if data is None:
        raise TypeError('data is None')

    info = ensure_persistence_class(data)

    ret = {}
    for field in info.fields:
        try:
            ret[field.field_name] = getattr(data, field.field_name)
        except AttributeError:
            pass
    return ret


def from_dict(data_cls: type[T], d: dict[str, Any]) -> T:
    """transform dictionary

    :param data_cls:
    :param d:
    :return:
    """
    info = ensure_persistence_class(data_cls)

    def get_or_raise(key):
        try:
            return d[key]
        except KeyError:
            pass

        raise KeyError(f'missing required field : {key}')

    init = {
        f.field_name: get_or_raise(f.field_name)
        for f in info.fields
        if f.init
    }

    try:
        ret = data_cls(**init)
    except TypeError:
        if hasattr(data_cls, 'from_dict'):
            return _from_dict_factory(data_cls, info, d)
        else:
            raise
    else:
        return _from_dict_builtin(ret, info, d)


def _from_dict_builtin(ret: T, info: PersistentClass, d: dict[str, Any]) -> T:
    for f in info.fields:
        if not f.init:
            try:
                v = d[f.field_name]
            except KeyError:
                pass
            else:
                setattr(ret, f.field_name, v)
    return ret


def _from_dict_factory(data_cls: type[T], info: PersistentClass, d: dict[str, Any]) -> T:
    kwargs = {}
    for f in info.fields:
        try:
            v = d[f.field_name]
        except KeyError:
            pass
        else:
            kwargs[f.field_name] = v

    ret = data_cls.from_dict(kwargs)
    if not isinstance(ret, data_cls):
        raise TypeError()
    return ret


def save(data: T, path: P) -> None:
    """save persistence **data** under **path**.

    :param data: persistence instance
    :param path: filepath
    """
    PickleHandler(type(data), path.parent).save_persistence(data, path)


def load(data_cls: type[T], path: P) -> T:
    """Load data as **data_cls** from **path**.

    :param data_cls: data class type
    :param path: filepath
    :return: persistence instance
    """
    return PickleHandler(data_cls, path.parent).load_persistence(path)


def load_by(data_cls: type[T], path: P, **kwargs) -> T:
    """Load **data_cls** from directory **path** with fields **kwargs**.

    :param data_cls: data class type
    :param path: directory
    :param kwargs: data fields
    :return: persistence instance
    """
    handler = PickleHandler(data_cls, path.parent)
    return handler.load_persistence(handler.filepath(None, **kwargs))


def filename(result: Union[T, type[T]], **kwargs) -> str:
    """Get data persistence filename.

    :param result:
    :param kwargs: overwrite fields.
    :return: filename
    :raise RuntimeError: *result*'s autoinc field not resolved, or filename field missing.
    :raise TypeError: wrong field.filename type.
    """
    cls_info = ensure_persistence_class(result)
    if isinstance(result, type):
        result = None

    if not cls_info.is_autoinc_field_resolved(result, **kwargs):
        af = cls_info.autoinc_field()
        raise RuntimeError(f'cannot generate filepath without autoinc field {af.field_name} keywords')

    name = cls_info.filename(result, **kwargs)

    return name


class PersistenceHandler(Generic[T], metaclass=abc.ABCMeta):
    """The handler for loading and saving persistence instance."""

    @property
    @abc.abstractmethod
    def persistence_class(self) -> type[T]:
        pass

    @property
    def persistence_info(self) -> PersistentClass[T]:
        """information for persistence class"""
        return ensure_persistence_class(self.persistence_class)

    @property
    @abc.abstractmethod
    def save_root(self) -> Path:
        """saving directory"""
        pass

    def filename(self, cache: Optional[T], **kwargs) -> str:
        """build filename for persistence instance.

        :param cache: persistence instance
        :param kwargs: overwrite field value in *result*.
        :return: file name of *result*, may contains '{}' if *result*'s autoinc field not resolved
        """
        cls_info = ensure_persistence_class(self.persistence_class)
        return cls_info.filename(cache, **kwargs)

    def filepath(self, cache: Optional[T], **kwargs) -> Path:
        """build filepath for persistence instance.

        :param cache: persistence instance
        :param kwargs: overwrite field value in *result*.
        :return: file path of *result*
        :raise RuntimeError: *result*'s autoinc field not resolved, or errors from :meth:`filename`
        """
        info = ensure_persistence_class(self.persistence_class)
        if not info.is_autoinc_field_resolved(cache, **kwargs):
            af = info.autoinc_field()
            raise AutoIncFieldNotResolvedError(cache, af,
                                               f'cannot generate filepath without autoinc field {af.field_name} keywords')

        name = self.filename(cache, **kwargs)
        return self.save_root / name

    def save_persistence(self, cache: T, path: Union[str, Path] = None) -> T:
        """save persistence *result* under **path**.

        :param cache:
        :param path: save path.
        :return: *result*. autoinc field will be resolved after saving.
        :raise AutoIncFieldNotResolvedError:
        """
        info = ensure_persistence_class(cache)

        if path is None:
            if not info.is_autoinc_field_resolved(cache):
                f = info.autoinc_field()
                found = [it for _, it in self.load_all(cache, **{f.field_name: '*'})]

                u = max([
                    value
                    for it in found
                    if isinstance(value := getattr(it, f.field_name, 0), int)
                ], default=-1) + 1

                setattr(cache, f.field_name, u)

            path = self.filepath(cache)

        else:
            if not info.is_autoinc_field_resolved(cache):
                raise AutoIncFieldNotResolvedError(cache, info.autoinc_field())

            if isinstance(path, str):
                path = Path(path)

        if path.is_dir():
            raise IsADirectoryError(str(path))

        path.parent.mkdir(parents=True, exist_ok=True)
        self._save_persistence(cache, path)
        return cache

    @abc.abstractmethod
    def _save_persistence(self, cache: T, path: Path):
        pass

    def load_persistence(self, path: Union[Path, T, dict[str, Any]]) -> T:
        """Load data as **data_cls** from **path**.

        :param path: load from path.
        :return: persistence instance
        :raise IsADirectoryError:
        """
        data_cls = self.persistence_class
        ensure_persistence_class(data_cls)

        if isinstance(path, data_cls):
            path = self.filepath(path)
        elif isinstance(path, dict):
            path = self.filepath(None, **path)
        elif isinstance(path, str):
            path = Path(path)

        if path.is_dir():
            raise IsADirectoryError(str(path))

        return self._load_persistence(path)

    @abc.abstractmethod
    def _load_persistence(self, path: Path) -> T:
        pass

    def load_all(self, cache: Optional[T], **kwargs) -> Iterable[tuple[Path, T]]:
        for file in self.save_root.glob(self.filename(cache, **kwargs)):
            yield file, self.load_persistence(file)


class PickleHandler(PersistenceHandler[T]):
    """

    Support field type: all python objects.

    """

    def __init__(self, data_cls: type[T], save_root: Path, ext: str = '.pkl'):
        ensure_persistence_class(data_cls)
        self._save_path = save_root
        self._data_cls = data_cls
        self._ext = ext

    @property
    def persistence_class(self) -> type[T]:
        return self._data_cls

    @property
    def save_root(self) -> Path:
        return self._save_path

    def filename(self, result: Optional[T], **kwargs) -> str:
        return super().filename(result, **kwargs) + self._ext

    def _save_persistence(self, result: T, path: Path):
        import pickle

        with path.open('wb') as file:
            pickle.dump(as_dict(result), file)

    def _load_persistence(self, path: Path) -> T:
        import pickle

        with path.open('rb') as file:
            ret = pickle.load(file)

        data_cls = self.persistence_class
        if isinstance(ret, dict):
            return from_dict(data_cls, ret)
        elif isinstance(ret, data_cls):  # for old persistent pickle file
            return ret
        else:
            raise TypeError(f'not a {data_cls.__name__} for cache {path} : {ret}')


class GzipHandler(PersistenceHandler[T]):
    """

    Support field type: all python objects.

    """

    def __init__(self, data_cls: type[T], save_root: Path, ext: str = '.pkl.gz',
                 compression: int = 9):
        ensure_persistence_class(data_cls)
        self._save_path = save_root
        self._data_cls = data_cls
        self._ext = ext
        self._cmp = compression

    @property
    def persistence_class(self) -> type[T]:
        return self._data_cls

    @property
    def save_root(self) -> Path:
        return self._save_path

    def filename(self, result: Optional[T], **kwargs) -> str:
        return super().filename(result, **kwargs) + self._ext

    def _save_persistence(self, result: T, path: Path):
        import pickle
        import gzip

        with gzip.open(path, 'wb', compresslevel=self._cmp) as file:
            pickle.dump(as_dict(result), file)

    def _load_persistence(self, path: Path) -> T:
        import pickle
        import gzip

        with gzip.open(path, 'rb') as file:
            ret = pickle.load(file)

        return from_dict(self.persistence_class, ret)


# ========================= #
# User-Specific for 2P data #
# ========================= #

class ETLConcatable(Generic[T], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def concat_etl(self, data: list[T]) -> T:
        pass


class PersistenceConcatError(Exception):
    pass


def validate_concat_etl_persistence(data: list[T],
                                    field_check: Optional[tuple[str, ...]] = None) -> None:
    """

    :param data:
    :param field_check: field name under persistence cls for checking
    :return:
    """
    #
    if not hasattr(data[0], 'exp_date') or not hasattr(data[0], 'animal'):
        raise RuntimeError('different dataset is not concatable')

    #
    for it in data:

        if not hasattr(it, 'plane_index'):
            raise AttributeError('not plane index field')

        if not isinstance(it.plane_index, int):
            raise TypeError('')

    if len(set([it.exp_date for it in data])) != 1:
        print(set([it.exp_date for it in data]))
        raise PersistenceConcatError('different exp date')

    if len(set([it.animal for it in data])) != 1:
        raise PersistenceConcatError('different animal')

    #
    if field_check is not None:
        for f in field_check:
            init = getattr(data[0], f)

            for it in data:
                check = getattr(it, f)

                if not is_iterable(init):
                    if check != init:
                        raise PersistenceConcatError(f' field:{f} not consistent')

                elif is_iterable(init) and isinstance(init, np.ndarray):
                    if not np.array_equal(init, check):
                        raise PersistenceConcatError(f' field:{f} not consistent')

                else:
                    for i, v in enumerate(init):
                        check_field = check
                        if check_field[i] != v:
                            raise PersistenceConcatError(f' field:{f} not consistent')
