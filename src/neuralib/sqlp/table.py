from __future__ import annotations

import abc
import typing
from typing import NamedTuple, Any, Optional, Generic, TYPE_CHECKING, Callable

from .annotation import *
from .literal import FOREIGN_POLICY, CONFLICT_POLICY

if TYPE_CHECKING:
    from .expr import SqlExpr, SqlField

__all__ = [
    'Field', 'table_name', 'table_field_names', 'table_fields', 'table_field',
    'table_primary_fields',
    'UniqueConstraint', 'table_unique_fields', 'make_unique_constraint',
    'ForeignConstraint', 'table_foreign_fields', 'table_foreign_field', 'make_foreign_constrain',
    'CheckConstraint', 'table_check_fields', 'table_check_field', 'make_check_constraint',

]

T = typing.TypeVar('T')
F = typing.TypeVar('F')
missing = object()


class Field(NamedTuple):
    """
    A SQL column, a field in a table.
    """
    table: type
    """associated table"""

    name: str
    """column/field name"""

    raw_type: type
    """origin field type"""

    sql_type: type
    """column/field type. Should be supported by SQL"""

    f_value: Any = missing
    """default value."""

    not_null: bool = True
    """Is it not NULL."""

    annotated: list[Any] = ()

    @property
    def table_name(self) -> str:
        return table_name(self.table)

    @property
    def has_default(self) -> bool:
        if self.f_value is not missing:
            return True

        if (primary := self.get_primary()) is not None:
            return primary.auto_increment

        return False

    @property
    def is_primary(self) -> bool:
        return self.get_primary() is not None

    def get_primary(self) -> PRIMARY | None:
        for a in self.annotated:
            if a == PRIMARY:
                return PRIMARY()
            elif isinstance(a, PRIMARY):
                return a
        return None

    @property
    def is_unique(self) -> bool:
        return self.get_unique() is not None

    def get_unique(self) -> UNIQUE | None:
        for a in self.annotated:
            if a == UNIQUE:
                return UNIQUE()
            elif isinstance(a, UNIQUE):
                return a
        return None

    def get_annotation(self, annotation_type: type[T]) -> T | None:
        for a in self.annotated:
            if a == annotation_type or isinstance(a, annotation_type):
                return a
        return None

    @typing.overload
    def __call__(self, data: type) -> SqlField:
        pass

    @typing.overload
    def __call__(self, data: Any) -> Any:
        pass

    def __call__(self, data):
        return getattr(data, self.name)

class Table(Generic[T], metaclass=abc.ABCMeta):
    """
    SQL table information.
    """

    table_type: type[T]
    """associated table"""

    table_name: str
    """name of the table."""

    @abc.abstractmethod
    def table_seq(self, instance: T, fields: list[str] = None) -> tuple[Any, ...]:
        """
        cast an instance as a tuple as SQL parameters.

        :param instance:
        :param fields:
        :return:
        """
        pass

    @abc.abstractmethod
    def table_new(self, *args) -> T:
        """create an instance."""
        pass

    @property
    def table_field_names(self) -> list[str]:
        """list of the name for each field in the table."""
        return [it.name for it in self.table_fields]

    @property
    @abc.abstractmethod
    def table_fields(self) -> list[Field]:
        """list fields in the table."""
        pass

    @property
    def table_primary_fields(self) -> list[Field]:
        """list of primary field in the table."""
        return [field for field in self.table_fields if field.is_primary]

    def table_field(self, name: str) -> Field:
        """
        Get field by the name in the table.

        :param name:
        :return:
        :raise RuntimeError: no such field.
        """
        for field in self.table_fields:
            if field.name == name:
                return field
        raise RuntimeError(f'{self.table_name} no such field {name}')

    @property
    @abc.abstractmethod
    def table_unique_fields(self) -> list[UniqueConstraint]:
        """get a list of the unique constraint in the table."""
        pass

    @property
    @abc.abstractmethod
    def table_foreign_fields(self) -> list[ForeignConstraint]:
        """get a list of the foreign constraint in the table."""
        pass

    @property
    @abc.abstractmethod
    def table_check_fields(self) -> dict[Optional[str], CheckConstraint]:
        """get a dict of the field constraint in the table."""
        pass


def table_class(cls: type[T]) -> Table[T]:
    return getattr(cls, '_sql_table')


def table_name(table: type[T]) -> str:
    """the name of the table."""
    return table_class(table).table_name


class UniqueConstraint(NamedTuple):
    name: str
    """constraint name"""

    table: type
    """associated table"""

    fields: list[str]
    """associated fields"""

    conflict: CONFLICT_POLICY | None


class ForeignConstraint(NamedTuple):
    """SQL foreign constraint."""

    name: str
    """constraint name"""

    table: type
    """associated table"""
    fields: list[str]
    """associated fields"""

    foreign_table: type
    """referred foreign table"""
    foreign_fields: list[str]
    """referred foreign fields"""

    on_update: FOREIGN_POLICY
    on_delete: FOREIGN_POLICY

    @property
    def table_name(self) -> str:
        return table_name(self.table)

    @property
    def foreign_table_name(self) -> str:
        return table_name(self.foreign_table)


class CheckConstraint(NamedTuple):
    """SQL check constraint"""
    name: str
    """constraint name"""

    table: type
    """associated table"""

    field: str | None
    """associated field's name."""

    expression: SqlExpr
    """checking expression"""


def table_field_names(table: type[T]) -> list[str]:
    """list of the name for each field in the table."""
    return table_class(table).table_field_names


def table_fields(table: type[T]) -> list[Field]:
    """list fields in the table."""
    return table_class(table).table_fields


def table_field(table: type[T], field: str) -> Field:
    """
    Get field by the name in the table.

    :param table:
    :param field:
    :return:
    :raise RuntimeError: no such field.
    """
    return table_class(table).table_field(field)


def table_primary_fields(table: type[T]) -> list[Field]:
    """list of the name for each primary field in the table."""
    return table_class(table).table_primary_fields


def table_unique_fields(table: type[T]) -> list[UniqueConstraint]:
    """list of the name for each unique field in the table."""
    return table_class(table).table_unique_fields


def table_foreign_fields(table: type[T]) -> list[ForeignConstraint]:
    """get a list of the foreign constraint in the table."""
    return table_class(table).table_foreign_fields


@typing.overload
def table_foreign_field(table: Callable) -> ForeignConstraint | None:
    pass


@typing.overload
def table_foreign_field(table: type[T], target: type[F]) -> ForeignConstraint | None:
    pass


def table_foreign_field(table: type[T] | Callable, target: type[F] = None) -> ForeignConstraint | None:
    """
    get the foreign constraint in the table that refer to the target table.

    :param table: table or a foreign constraint function/property (the function decorated by @foreign)
    :param target: refer table
    :return: foreign constraint.
    """
    if isinstance(table, type) and isinstance(target, type):
        for constraint in table_foreign_fields(table):
            if constraint.foreign_table == target:
                return constraint
        return None
    elif target is None:
        constraint = getattr(table, '_sql_foreign', None)
        if isinstance(constraint, ForeignConstraint):
            return constraint
        return None
    else:
        raise TypeError(repr(table))


def table_check_fields(table: type[T]) -> dict[Optional[str], CheckConstraint]:
    """get a dict of the field constraint in the table."""
    return table_class(table).table_check_fields


def table_check_field(table: type[T], field: Optional[str]) -> Optional[CheckConstraint]:
    """get the check constrain of a field in the table."""
    return table_class(table).table_check_fields.get(field, None)


def make_foreign_constrain(table: Table,
                           prop: callable,
                           fields: list,
                           update: FOREIGN_POLICY,
                           delete: FOREIGN_POLICY) -> ForeignConstraint:
    from .expr import SqlField

    foreign_fields = []

    if len(fields) == 0:
        raise RuntimeError('empty fields')
    elif isinstance(fields[0], str):
        if not all([isinstance(it, str) for it in fields]):
            raise TypeError()

        foreign_table = table.table_type
        foreign_fields.extend(fields)
    elif len(fields) == 1 and isinstance(fields[0], type):
        foreign_table = fields[0]
        foreign_fields = [it.name for it in table_primary_fields(foreign_table)]
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

    ret = prop(table.table_type)
    if not isinstance(ret, tuple):
        ret = [ret]

    self_fields = []
    for field in ret:
        if isinstance(field, SqlField):
            if table.table_type != field.table:
                raise RuntimeError()
            self_fields.append(field.name)
        else:
            raise TypeError()

    return ForeignConstraint(prop.__name__, table.table_type, self_fields, foreign_table, foreign_fields, update, delete)


def make_check_constraint(table: Table, prop: callable, field: str | None) -> CheckConstraint:
    from .expr import wrap
    ret = wrap(prop(table.table_type))
    return CheckConstraint(prop.__name__, table.table_type, field, ret)


def make_unique_constraint(table: Table, prop: callable, conflict: CONFLICT_POLICY = None) -> UniqueConstraint:
    from .expr import SqlField

    ret = prop(table.table_type)
    if not isinstance(ret, tuple):
        ret = [ret]

    fields = []
    for field in ret:
        if isinstance(field, SqlField):
            if table.table_type != field.table:
                raise RuntimeError()
            fields.append(field.name)
        else:
            raise TypeError()

    return UniqueConstraint(prop.__name__, table.table_type, fields, conflict)
