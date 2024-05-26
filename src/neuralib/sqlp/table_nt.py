from typing import TypeVar, Generic, Any, NamedTuple, Annotated, Optional

from .expr import SqlField
from .literal import FOREIGN_POLICY
from .table import Table, ForeignConstraint, Field, missing, PRIMARY, UNIQUE, CheckConstraint
from .util import resolve_field_type, cast_from_sql, cast_to_sql

__all__ = ['named_tuple_table_class']

T = TypeVar('T')


def named_tuple_table_class(cls):
    """
    A deceorator that deceorate a NamedTuple to be a SQL table.

    Declare a table

    >>> @named_tuple_table_class
    ... class Example(NamedTuple):
    ...     a: Annotated[str, PRIMARY]  # primary key
    ...     b: Annotated[str, UNIQUE]   # unique key
    ...     c: Optional[str]            # nullable key

    """
    ret = NamedTupleTable(cls)
    setattr(cls, '_sql_table', ret)
    return cls


class NamedTupleField:
    def __init__(self, field: Field, raw_type: type):
        self.field = field
        self.raw_type = raw_type

    @property
    def name(self) -> str:
        return self.field.name

    @property
    def sql_type(self):
        return self.field.f_type

    def cast_to_sql(self, value):
        return cast_to_sql(self.raw_type, self.sql_type, value)

    def cast_from_sql(self, value):
        return cast_from_sql(self.raw_type, self.sql_type, value)


class NamedTupleTable(Table[T], Generic[T]):
    """
    SQL table information for a NamedTuple class that decorated by named_tuple_table_class.
    """

    def __init__(self, table_type: type[T]):
        import typing

        if not hasattr(table_type, '_fields'):
            raise RuntimeError(f'not a NamedTuple {table_type.__name__}')

        self.table_type = table_type
        self._fields: list[NamedTupleField] = []
        self._primary: list[str] = []
        self._unique: list[str] = []
        self._foreign: list[ForeignConstraint] = []
        self._check: dict[Optional[str], CheckConstraint] = {}

        field_types = typing.get_type_hints(table_type, include_extras=False)
        for i, name in enumerate(getattr(table_type, '_fields')):
            r_type, f_type, not_null = resolve_field_type(field_types[name])
            f_value = table_type._field_defaults.get(name, missing)
            field = Field(table_type, name, f_type, f_value, not_null)
            setattr(table_type, name, TableFieldDescriptor(i, field))
            self._fields.append(NamedTupleField(field, r_type))

        field_types = typing.get_type_hints(table_type, include_extras=True)
        for field, f_type in field_types.items():
            if typing.get_origin(f_type) == typing.Annotated:
                a = typing.get_args(f_type)
                if PRIMARY in a:
                    self._primary.append(field)
                elif UNIQUE in a:
                    self._unique.append(field)

        for attr in dir(table_type):
            if callable(attr_value := getattr(table_type, attr)):
                if (foreign := getattr(attr_value, '_sql_foreign', None)) is not None:
                    self._foreign.append(make_foreign_constrain(self, attr_value, *foreign))
                if (check := getattr(attr_value, '_sql_check', missing)) is not missing:
                    check = make_check_constraint(self, attr_value, check)
                    self._check[check.field] = check

    @property
    def table_name(self) -> str:
        return self.table_type.__name__

    def table_seq(self, instance: T) -> tuple[Any, ...]:
        _args = []
        for field, arg in zip(self._fields, instance):
            _args.append(field.cast_to_sql(arg))
        return tuple(_args)

    def table_dict(self, instance: T, *, sql_type: bool = True) -> dict[str, Any]:
        ret = {}
        for field, arg in zip(self._fields, instance):
            if sql_type:
                ret[field.name] = field.cast_to_sql(arg)
            else:
                ret[field.name] = arg
        return ret

    def table_new(self, *args) -> T:
        _args = []
        for field, arg in zip(self._fields, args):
            _args.append(field.cast_from_sql(arg))
        return self.table_type(*_args)

    @property
    def table_fields(self) -> list[Field]:
        return [it.field for it in self._fields]

    @property
    def table_primary_field_names(self) -> list[str]:
        return list(self._primary)

    @property
    def table_unique_field_names(self) -> list[str]:
        return list(self._unique)

    @property
    def table_foreign_fields(self) -> list[ForeignConstraint]:
        return list(self._foreign)

    @property
    def table_check_fields(self) -> dict[Optional[str], CheckConstraint]:
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


def make_foreign_constrain(table: NamedTupleTable,
                           prop: callable,
                           fields: list,
                           update: FOREIGN_POLICY,
                           delete: FOREIGN_POLICY) -> ForeignConstraint:
    foreign_fields = []

    if len(fields) == 0:
        raise RuntimeError('empty fields')
    elif isinstance(fields[0], str):
        if not all([isinstance(it, str) for it in fields]):
            raise TypeError()

        foreign_table = table.table_type
        foreign_fields.extend(fields)
    elif len(fields) == 1 and isinstance(fields[0], type):
        from .table import table_primary_field_names
        foreign_table = fields[0]
        foreign_fields = table_primary_field_names(foreign_table)
    else:
        foreign_table = None

        for field in fields:
            if isinstance(field, SqlField):
                if foreign_table is None:
                    foreign_table = field.table
                elif foreign_table != field.table:
                    raise RuntimeError()

                foreign_fields.append(field.name)
            else:
                raise TypeError()

    self_fields = []
    ret = prop(table.table_type)
    if not isinstance(ret, tuple):
        ret = [ret]

    for field in ret:
        if isinstance(field, SqlField):
            if table.table_type != field.table:
                raise RuntimeError()
            self_fields.append(field.name)
        else:
            raise TypeError()

    return ForeignConstraint(prop.__name__, table.table_type, self_fields, foreign_table, foreign_fields, update,
                             delete)


def make_check_constraint(table: NamedTupleTable, prop: callable, field: Optional[str]) -> CheckConstraint:
    from .expr import wrap
    ret = wrap(prop(table.table_type))
    return CheckConstraint(table.table_type, field, ret)
