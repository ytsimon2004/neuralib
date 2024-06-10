from __future__ import annotations

import datetime
import operator
import re
from pathlib import Path
from typing import TypeVar, Iterable, overload, TYPE_CHECKING, Any

from neuralib.util.table import rich_table
from .expr import SqlExpr, SqlField

if TYPE_CHECKING:
    from .stat import Cursor

__all__ = [
    'str_to_datetime',
    'datetime_to_str',
    'take',
    'infer_eq',
    'infer_cmp',
    'infer_in',
    'resolve_field_type',
    'cast_to_sql',
    'cast_from_sql',
    'get_fields_from_schema',
    'map_foreign',
    'pull_foreign',
    'rich_sql_table'
]

T = TypeVar('T')
V = TypeVar('V')


def str_to_datetime(t: str) -> datetime.datetime:
    return datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')


def datetime_to_str(t: datetime.datetime) -> str:
    return t.strftime('%Y-%m-%d %H:%M:%S')


@overload
def take(index: int, coll: Cursor | Iterable[tuple[T, ...]]) -> list[T]:
    pass


@overload
def take(index: type[V], coll: Cursor | Iterable[tuple[T, ...]]) -> list[V]:
    pass


@overload
def take(index: tuple[int, ...], coll: Cursor | Iterable[tuple[T, ...]]) -> list[tuple[T, ...]]:
    pass


@overload
def take(index: tuple[V], coll: Cursor | Iterable[T]) -> list[tuple[V]]:
    pass


@overload
def take(index: V, coll: Cursor | Iterable[T]) -> list[V]:
    pass


def take(index, coll: Cursor | Iterable):
    """
    A help function that compose itemgetter and mapping functions.

    >>> @named_tuple_table_class
    ... class A:
    ...     a: int
    ...     b: str
    >>> take(0, [(0, 'a'), (1, 'b')])
    [0, 1]
    >>> take(A.a, [A(0, 'a'), A(1, 'b')])
    [0, 1]


    :param index:
    :param coll:
    :return:
    """
    if isinstance(index, type):
        return list(map(lambda it: index(*it), coll))

    if isinstance(index, int):
        return list(map(operator.itemgetter(index), coll))

    if isinstance(index, tuple) and all([isinstance(it, int) for it in index]):
        def _index(item):
            return tuple([item[it] for it in index])

        return list(map(_index, coll))

    from .table import table_field_names
    from .stat import Cursor

    if isinstance(index, SqlField):
        if isinstance(coll, Cursor):
            index = coll.headers.index(index.field.name)
        else:
            fields = table_field_names(index.field.table)
            index = fields.index(index.field.name)

        return list(map(operator.itemgetter(index), coll))

    if isinstance(index, tuple):
        index = cast(tuple[SqlField], index)
        if isinstance(coll, Cursor):
            index = tuple([coll.headers.index(it.field.name) for it in index])
        else:
            fields = table_field_names(index[0].field.table)
            index = tuple([fields.index(it.field.name) for it in index])

        return take(index, coll)

    raise TypeError()


def infer_eq(x: T, v: T | str, *, prepend: str = '', append: str = '') -> SqlExpr | None:
    """
    A help function to make a SQL ``=`` expression.

    >>> infer_eq(A.a, 1) # doctest: SKIP
    A.a = 1
    >>> infer_eq(A.a, '!1') # doctest: SKIP
    A.a != 1
    >>> infer_eq(A.a, '1%') # doctest: SKIP
    A.a LIKE '1%'

    :param x:
    :param v:
    :param prepend:
    :param append:
    :return:
    """
    if not isinstance(x, SqlField):
        raise TypeError()

    if v is None:
        return None
    if isinstance(v, SqlExpr):
        return v

    invert = False
    if isinstance(v, str) and v.startswith('!'):
        invert = True
        v = v[1:]

    if isinstance(v, str) and '%' in v:
        from .func_stat import like, not_like
        return not_like(x, v) if invert else like(x, v)

    if isinstance(v, str) and (prepend == '%' or append == '%'):
        from .func_stat import like, not_like
        v = prepend + v + append
        return not_like(x, v) if invert else like(x, v)

    return x != v if invert else x == v


def infer_cmp(x: T, v: T | str | range | slice) -> SqlExpr | None:
    """
    A help function to make a SQL comparison expression.

    >>> infer_cmp(A.a, range(0, 10))  # doctest: SKIP
    A.a BETWEEN 0 AND 9
    >>> infer_cmp(A.a, slice(0, 10))  # doctest: SKIP
    A.a BETWEEN 0 AND 10
    >>> infer_cmp(A.a, '<10')  # doctest: SKIP
    A.a < 10
    >>> infer_cmp(A.a, 10)  # doctest: SKIP
    A.a = 10

    :param x:
    :param v:
    :return:
    """
    if not isinstance(x, SqlField):
        raise TypeError()

    if v is None:
        return None
    if isinstance(v, SqlExpr):
        return v
    if isinstance(v, (int, float)):
        return x == v
    if isinstance(v, (range, slice)):
        return infer_in(x, v)

    if '%' in v:
        from .func_stat import like
        return like(x, v)

    if v.startswith('<='):
        return x <= float(v[2:])
    elif v.startswith('>='):
        return x >= float(v[2:])
    elif v.startswith('<'):
        return x < float(v[1:])
    elif v.startswith('>'):
        return x > float(v[1:])

    invert = False
    if v.startswith('!'):
        invert = True
        v = v[1:]

    return x != v if invert else x == v


def infer_in(x: T, v: T | str | list[str] | slice | range) -> SqlExpr | None:
    """
    A help function to make a SQL containing expression.

    >>> infer_in(A.a, '1')  # doctest: SKIP
    A.a == '1'
    >>> infer_in(A.a, range(0, 10))  # doctest: SKIP
    A.a BETWEEN 0 AND 9
    >>> infer_in(A.a, ['a', 'b'])  # doctest: SKIP
    A.a IN ('a', 'b')

    :param x:
    :param v:
    :return:
    """
    if not isinstance(x, SqlField):
        raise TypeError()

    if v is None:
        return None
    if isinstance(v, SqlExpr):
        return v

    if isinstance(v, str):
        return infer_eq(x, v)
    if isinstance(v, (range, slice)):
        return x.between(v)
    return x.contains(v)


def resolve_field_type(f_type: type) -> tuple[type, type, bool]:
    """

    SQL primary types:

    * bool: BOOLEAN
    * int: INT
    * float: FLOAT
    * str: TEXT
    * bytes: BLOB
    * datetime.date: DATETIME
    * datetime.datetime: DATETIME

    Python type mapping

    * `T|None`: `resolve_field_type(T)` null-able
    * `T|V` : supported not
    * `Literal`: `str`
    * `Path`: `str`

    :param f_type:
    :return: (raw_type, sql_type, not_null)
    """
    import typing

    sql_type = f_type
    o = typing.get_origin(f_type)

    if o == typing.Annotated:
        return resolve_field_type(typing.get_args(f_type)[0])

    elif o == typing.Union:
        a = typing.get_args(f_type)
        if len(a) == 2:
            try:
                i = a.index(type(None))
            except ValueError as e:
                raise RuntimeError('Union type is not supported now') from e
            else:
                f_type = a[1 - i]
                _, sql_type, _ = resolve_field_type(f_type)
                return f_type, sql_type, False

    elif o == typing.Literal:
        return str, str, True

    elif f_type == Path:
        return f_type, str, True

    return f_type, sql_type, True


def cast_to_sql(raw_type: type[T], sql_type: type[V], value: T) -> V:
    if value is None:
        return None
    if sql_type == str:
        return str(value)
    if raw_type == Any:
        return value
    return value


def cast_from_sql(raw_type: type[T], sql_type: type[V], value: V) -> T:
    if value is None:
        return None
    if raw_type == Any:
        return value
    if sql_type in (int, float, bool):
        return value
    if raw_type == datetime.datetime and isinstance(value, str):
        return datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
    if raw_type == datetime.date and isinstance(value, str):
        return datetime.datetime.strptime(value, '%Y-%m-%d').date()
    if raw_type == sql_type:
        return value
    if raw_type == Path:
        return Path(value)
    if callable(raw_type):
        return raw_type(value)
    return value


def get_fields_from_schema(schema: str) -> list[str]:
    schema = schema[schema.index('(') + 1:]
    try:
        schema = schema[:schema.rfind(')')]
    except ValueError:
        pass

    schema = re.sub(r'\(.+?\)', '', schema.strip())

    found = []
    for field in schema.split(','):
        field = field.strip().split(' ')
        if field[0] in ('FOREIGN', 'UNIQUE', 'PRIMARY', 'CHECK'):
            break

        field = field[0]
        if len(field):
            if field.startswith('[') and field.endswith(']'):
                field = field[1:-1]
            found.append(field)

    return found


def map_foreign(value: T, foreign: type[V]) -> Cursor[V]:
    """
    Let a table ``T`` with a foreign constraint refer to table ``V``,
    map a ``T`` data to the ``V`` data.

    :param value:
    :param foreign: a foreign constraint
    :return:
    """
    from .table import table_foreign_field
    from .stat_start import select_from

    table = type(value)
    if (constraint := table_foreign_field(table, foreign)) is None:
        raise RuntimeError(f'not a foreign constraint : {foreign}')

    # SELECT * FROM V
    # WHERE AND*([V.field == t.field for field in constraint])
    return select_from(constraint.foreign_table).where(*[
        getattr(constraint.foreign_table, f) == getattr(value, t)
        for (t, f) in zip(constraint.fields, constraint.foreign_fields)
    ]).submit()


def pull_foreign(target: type[T], foreign: V) -> Cursor[T]:
    """
    Let a table ``T`` with a foreign constraint refer to table ``V``,
    pull ``T`` data from a ``V`` data.

    :param target: target table ``T``
    :param foreign: a foreign data ``V`` referred to.
    :return:
    """
    from .table import table_foreign_field
    from .stat_start import select_from

    if (constraint := table_foreign_field(target, type(foreign))) is None:
        raise RuntimeError(f'not a foreign constraint')

    # SELECT * FROM T
    # WHERE AND*([T.field == v.field for field in constraint])
    return select_from(constraint.table).where(*[
        getattr(constraint.table, t) == getattr(foreign, f)
        for (t, f) in zip(constraint.fields, constraint.foreign_fields)
    ]).submit()


def rich_sql_table(table: type[T], value: list[T]):
    from .table import table_field_names
    with rich_table(*table_field_names(table)) as _table:
        for _value in value:
            _table(*_value)
