import functools
import typing

from .annotation import *
from .table import *
from .table import Table, missing
from .util import resolve_field_type, cast_from_sql, cast_to_sql

__all__ = ['named_tuple_table_class']

T = typing.TypeVar('T')


def named_tuple_table_class(cls):
    """
    A deceorator that deceorate a NamedTuple to be a SQL table.

    Declare a table

    >>> @named_tuple_table_class
    ... class Example(typing.NamedTuple):
    ...     a: typing.Annotated[str, PRIMARY]  # primary key
    ...     b: typing.Annotated[str, UNIQUE]   # unique key
    ...     c: typing.Optional[str]            # nullable key

    """
    ret = NamedTupleTable(cls)
    setattr(cls, '_sql_table', ret)
    return cls


class NamedTupleTable(Table[T], typing.Generic[T]):
    """
    SQL table information for a NamedTuple class that decorated by named_tuple_table_class.
    """

    def __init__(self, table_type: type[T]):
        import typing

        if not hasattr(table_type, '_fields'):
            raise RuntimeError(f'not a NamedTuple {table_type.__name__}')

        self.table_type = table_type
        self._fields: list[Field] = []
        self._unique: list[UniqueConstraint] = []
        self._foreign: list[ForeignConstraint] = []
        self._check: dict[typing.Optional[str], CheckConstraint] = {}

        field_types = typing.get_type_hints(table_type, include_extras=True)
        for i, name in enumerate(getattr(table_type, '_fields')):
            field = self.__setup_column_constraint(table_type, i, name, field_types[name])

            if (constraint := field.get_unique()) is not None:
                self._unique.append(UniqueConstraint(field.name, table_type, [field.name], constraint.conflict))

        self.__setup_table_constraint(table_type)

    def __setup_column_constraint(self, table_type: type[T], i: int, attr_name: str, attr_type) -> Field:
        f_value_missing = missing

        attr_annotations = []
        if typing.get_origin(attr_type) == typing.Annotated:
            attr_annotations = typing.get_args(attr_type)[1:]
            if CURRENT_DATE in attr_annotations:
                f_value_missing = CURRENT_DATE
            elif CURRENT_TIME in attr_annotations:
                f_value_missing = CURRENT_TIME
            elif CURRENT_TIMESTAMP in attr_annotations:
                f_value_missing = CURRENT_TIMESTAMP

        r_type, f_type, not_null = resolve_field_type(attr_type)
        f_value = table_type._field_defaults.get(attr_name, f_value_missing)
        field = Field(table_type, attr_name, r_type, f_type, f_value, not_null, attr_annotations)
        setattr(table_type, attr_name, TableFieldDescriptor(i, field))
        self._fields.append(field)
        return field

    def __setup_table_constraint(self, table_type: type[T]):
        for attr in dir(table_type):
            if callable(attr_value := getattr(table_type, attr)):
                if (constraint := getattr(attr_value, '_sql_foreign', None)) is not None:
                    constraint = make_foreign_constrain(self, attr_value, *constraint)
                    self._foreign.append(constraint)
                    setattr(attr_value, '_sql_foreign', constraint)
                if (constraint := getattr(attr_value, '_sql_check', missing)) is not missing:
                    constraint = make_check_constraint(self, attr_value, *constraint)
                    self._check[constraint.field] = constraint
                    setattr(attr_value, '_sql_check', constraint)
                if (constraint := getattr(attr_value, '_sql_unique', missing)) is not missing:
                    constraint = make_unique_constraint(self, attr_value, *constraint)
                    self._unique.append(constraint)
                    setattr(attr_value, '_sql_unique', constraint)
            elif isinstance(attr_value, property):
                setattr(table_type, attr, setup_table_property(attr_value, table_type))

    @property
    def table_name(self) -> str:
        return self.table_type.__name__

    def table_seq(self, instance: T, fields: list[str] = None) -> tuple[typing.Any, ...]:
        _args = []
        for field, arg in zip(self._fields, instance):
            if field is None or field.name in fields:
                _args.append(cast_to_sql(field.raw_type, field.sql_type, arg))
        return tuple(_args)

    def table_new(self, *args) -> T:
        _args = []
        for field, arg in zip(self._fields, args):
            _args.append(cast_from_sql(field.raw_type, field.sql_type, arg))
        return self.table_type(*_args)

    @property
    def table_fields(self) -> list[Field]:
        return list(self._fields)

    @property
    def table_unique_fields(self) -> list[UniqueConstraint]:
        return list(self._unique)

    @property
    def table_foreign_fields(self) -> list[ForeignConstraint]:
        return list(self._foreign)

    @property
    def table_check_fields(self) -> dict[typing.Optional[str], CheckConstraint]:
        return dict(self._check)


class TableFieldDescriptor:
    __slots__ = '__index', '__field'

    def __init__(self, index: int, field: Field):
        self.__index = index
        self.__field = field

    def __get__(self, instance, owner):
        if instance is None:
            from .expr import SqlField
            return SqlField(self.__field)
        else:
            return instance[self.__index]

    def __str__(self):
        return self.__field.name


def setup_table_property(prop: property, table_type: type) -> property:
    getter = wrap_property_getter(prop.fget, table_type)
    setter = prop.fset
    deleter = prop.fdel
    return property(getter, setter, deleter)


def wrap_property_getter(getter, table_type: type):
    from .expr import SqlExpr

    @functools.wraps(getter)
    def getter_wrapper(self):
        if isinstance(self, type):
            return getter(self)

        ret = getter(self)
        if isinstance(ret, SqlExpr):
            ret = ret.__sql_eval__(self)
        return ret

    getter_wrapper._sql_owner = table_type
    return getter_wrapper
