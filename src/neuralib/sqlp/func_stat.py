"""
SQL help functions.
"""
from __future__ import annotations

from collections.abc import Collection
from typing import overload, Any, TypeVar, TYPE_CHECKING

from . import expr

if TYPE_CHECKING:
    from .stat import SqlStat, SqlSelectStat

__all__ = [
    'TRUE', 'FALSE', 'NULL', 'ROWID',
    'literal', 'wrap', 'alias', 'cast', 'case', 'exists', 'asc', 'desc', 'nulls_first', 'nulls_last', 'concat',
    'and_', 'or_',
    'like', 'not_like', 'glob', 'contains', 'not_contains', 'between', 'not_between', 'is_null', 'is_not_null',
    'excluded', 'with_common_table', 'fields'
]

T = TypeVar('T')

TRUE = expr.SqlLiteral.SQL_TRUE
FALSE = expr.SqlLiteral.SQL_FALSE
NULL = expr.SqlLiteral.SQL_NULL
ROWID = expr.SqlLiteral('rowid')


def literal(x: str) -> expr.SqlExpr:
    return expr.SqlLiteral(x)


def wrap(x) -> expr.SqlExpr:
    """wrap *x* as an SQL expression."""
    return expr.wrap(x)


@overload
def alias(x: str, name: str) -> Any:
    pass


@overload
def alias(x: type[T], name: str) -> type[T]:
    pass


@overload
def alias(x: SqlStat[T], name: str) -> type[T]:
    pass


def alias(x, name: str):
    """
    ..  code-block::SQL

        :x AS :name

    """
    if isinstance(x, type):
        return expr.SqlAlias(x, name)
    return expr.SqlAlias(expr.wrap(x), name)


def cast(t: type[T], x) -> T:
    """
    ..  code-block::SQL

        CAST(:x AS :t)

    """
    return expr.SqlCastOper(t.__name__, expr.wrap(x))


def case(x=None) -> expr.SqlCaseExpr:
    """
    https://www.sqlite.org/lang_expr.html#the_case_expression

    .. code-block::SQL

        CASE :x
            ...

    :param x:
    :return:
    """
    return expr.SqlCaseExpr(None if x is None else expr.wrap(x))


@overload
def exists(x: SqlStat) -> expr.SqlExpr:
    pass


@overload
def exists(x: type, *where: bool | expr.SqlExpr) -> expr.SqlExpr:
    pass


def exists(x, *where: bool | expr.SqlExpr) -> expr.SqlExpr:
    """

    https://www.sqlite.org/lang_expr.html#the_exists_operator

    >>> exists(A, A.a == 1) # equivalent below
    >>> exists(select_from(1, from_table=A).where(A.a == 1))

    :param x:
    :param where:
    :return:
    """
    if isinstance(x, type):
        from .stat_start import select_from
        x = select_from(1, from_table=x).where(*where)
    return expr.SqlExistsOper('EXISTS', x)


def asc(x) -> expr.SqlExpr:
    """
    ascending ordering Used by **ORDER BY**.

    ..  code-block::SQL

        :x ASC


    """
    return expr.SqlOrderOper('ASC', expr.wrap(x))


def desc(x) -> expr.SqlExpr:
    """
    descending ordering used by **ORDER BY**.

    ..  code-block::SQL

        :x DESC

    """
    return expr.SqlOrderOper('DESC', expr.wrap(x))


def nulls_first(x) -> expr.SqlExpr:
    """
    order null first used by **ORDER BY**.

    ..  code-block::SQL

        :x NULLS FIRST

    """
    return expr.SqlOrderOper('NULLS FIRST', expr.wrap(x))


def nulls_last(x) -> expr.SqlExpr:
    """
    order null last used by **ORDER BY**.

    ..  code-block::SQL

        :x NULLS LAST

    """
    return expr.SqlOrderOper('NULLS LAST', expr.wrap(x))


def concat(*x) -> expr.SqlExpr:
    """
    concatenate strings.

    ..  code-block::SQL

        :x[0] || :x[1] || ...

    """
    return expr.SqlConcatOper(expr.wrap_seq(*x))


def and_(*other) -> expr.SqlExpr:
    """
    "AND" SQL expressions.

    ..  code-block::SQL

        other[0] AND other[1] AND ...

    """
    if len(other) == 0:
        raise RuntimeError()
    elif len(other) == 1:
        return expr.wrap(other[0])
    return expr.SqlVarArgOper('AND', expr.wrap_seq(*other))


def or_(*other) -> expr.SqlExpr:
    """
    "OR" SQL expressions.

    ..  code-block::SQL

        other[0] OR other[1] OR ...

    """
    if len(other) == 0:
        raise RuntimeError()
    elif len(other) == 1:
        return expr.wrap(other[0])
    return expr.SqlVarArgOper('OR', expr.wrap_seq(*other))


def like(x, s) -> expr.SqlCompareOper:
    """
    ..  code-block::SQL

        :x LIKE :s


    :param x:
    :param s:
    :return:
    """
    return expr.wrap(x).like(s)


def not_like(x, s) -> expr.SqlCompareOper:
    """
    ..  code-block::SQL

        :x NOT LIKE :s


    :param x:
    :param s:
    :return:
    """
    return expr.wrap(x).not_like(s)


def glob(x, s) -> expr.SqlCompareOper:
    """
    ..  code-block::SQL

        :x GLOB :s


    :param x:
    :param s:
    :return:
    """
    return expr.wrap(x).glob(s)


def contains(x, coll) -> expr.SqlCompareOper:
    """
    ..  code-block::SQL

        :x IN :COLL


    :param x:
    :param coll: a sequence or a select statement.
    :return:
    """
    from .stat import SqlStat
    if not isinstance(coll, (tuple, list, SqlStat)):
        coll = [coll]
    return expr.wrap(x).contains(coll)


def not_contains(x, coll) -> expr.SqlCompareOper:
    """
    ..  code-block::SQL

        :x NOT IN :COLL


    :param x:
    :param coll: a sequence or a select statement.
    :return:
    """
    from .stat import SqlStat
    if not isinstance(coll, (tuple, list, SqlStat)):
        coll = [coll]
    return expr.wrap(x).not_contains(coll)


def between(x, *value) -> expr.SqlCompareOper:
    """
    https://www.sqlite.org/lang_expr.html#the_between_operator

    ..  code-block::SQL

        :x BETWEEN :value[0] AND :value[1]


    :param x:
    :param value: two value, or a range, a slice.
    :return:
    """
    return expr.wrap(x).between(*value)


def not_between(x, *value) -> expr.SqlCompareOper:
    """
    https://www.sqlite.org/lang_expr.html#the_between_operator

    ..  code-block::SQL

        :x NOT BETWEEN :value[0] AND :value[1]


    :param x:
    :param value: two value, or a range, a slice.
    :return:
    """
    return expr.wrap(x).not_between(*value)


def is_null(x) -> expr.SqlCompareOper:
    """
    ..  code-block::SQL

        :x IS NULL

    """
    return expr.wrap(x).is_null()


def is_not_null(x) -> expr.SqlCompareOper:
    """
    ..  code-block::SQL

        :x IS NOT NULL

    """
    return expr.wrap(x).is_not_null()


def excluded(t: type[T]) -> type[T]:
    """https://www.sqlite.org/lang_upsert.html"""
    return expr.SqlAlias(t, 'excluded')


@overload
def with_common_table(name: str, select: SqlSelectStat) -> expr.SqlCteExpr:
    pass


@overload
def with_common_table(name: type[T], select: SqlSelectStat) -> type[T]:
    pass


def with_common_table(name: str, select: SqlSelectStat):
    if isinstance(name, type):
        name = name.__name__
    return expr.SqlCteExpr(name, select)


# noinspection PyShadowingNames
def fields(table: type[T], *,
           primary: bool = None,
           has_default: bool = None,
           excluded: Collection[str] = None) -> tuple[expr.SqlField, ...]:
    from .table import table_fields
    fields = table_fields(table)

    if primary is not None:
        fields = [it for it in fields if it.is_primary == primary]
    if has_default is not None:
        fields = [it for it in fields if it.has_default == has_default]
    if excluded is not None:
        fields = [it for it in fields if it.name not in excluded]

    return tuple([it(table) for it in fields])
