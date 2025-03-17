import datetime
from typing import overload, Any, TypeVar, Union

from .annotation import *
from .expr import *
from .expr import use_table_first, SqlRemovePlaceHolder
from .literal import UPDATE_POLICY
from .stat import *
from .table import *

__all__ = [
    'create_table',
    'insert_into',
    'replace_into',
    'select_from',
    'update',
    'delete_from'
]

T = TypeVar('T')
S = TypeVar('S')


@overload
def select_from(table: type[T], *, distinct: bool = False) -> SqlSelectStat[T]:
    """
    >>> select_from(Table) # SELECT * FROM Table
    """
    pass


@overload
def select_from(table: SqlCteExpr, *, distinct: bool = False) -> SqlSelectStat[tuple]:
    pass


@overload
def select_from(*field, distinct: bool = False,
                from_table: Union[str, type, SqlAlias, SqlSelectStat] = None) -> SqlSelectStat[tuple]:
    """
    >>> select_from('a', 'b') # SELECT a, b FROM Table
    """
    pass


def select_from(*args, distinct: bool = False,
                from_table: Union[str, type, SqlAlias, SqlSelectStat] = None) -> SqlSelectStat:
    """
    ``SELECT``: https://www.sqlite.org/lang_select.html

    Select all fields from a table

    >>> select_from(A).build() # doctest: SKIP
    SELECT * FROM A
    >>> select_from(A).fetchall() # doctest: SKIP
    [A(...), A(...), ...]

    Select subset of fields from A

    >>> select_from(A.a, A.b).build() # doctest: SKIP
    SELECT A.a, A.b FROM A
    >>> select_from(A.a, A.b).fetchall() # doctest: SKIP
    [('a', 1), ('b', 2), ...]

    With a literal value

    >>> select_from(A.a, 0).build() # doctest: SKIP
    SELECT A.a, 0 FROM A
    >>> select_from(A.a, 0).fetchall() # doctest: SKIP
    [('a', 0), ('b', 0), ...]

    With SQL functions

    >>> select_from(A.a, count()).build() # doctest: SKIP
    SELECT A.a, COUNT(*) FROM A

    Use table alias

    >>> a = alias(A, 'a') # doctest: SKIP
    >>> select_from(a.a).build() # doctest: SKIP
    SELECT a.a from A a

    join other tables

    >>> select_from(A.a, B.b).join(A.c == B.c).build() # doctest: SKIP
    SELECT A.a, B.b FROM A JOIN B ON A.c = B.c

    **features supporting**

    * `SELECT DISTINCT`
    * `FROM`
    * `WHERE`
    * `GROUP BY`
    * `HAVING`
    * `WINDOW`
    * compound-operator: `UNION [ALL]`, `INTERSECT` and `EXCEPT`
    * `ORDER BY`
    * `LIMIT [OFFSET]`

    **features not supporting**

    * `WITH [RECURSIVE]`
    * `SELECT ALL`
    * `VALUES`

    :param args:
    :param distinct:
    :param from_table:
    :return:
    """
    pre_stat = ['SELECT']
    if distinct:
        pre_stat.append('DISTINCT')

    if len(args) == 1 and isinstance(table := args[0], type):
        self = SqlSelectStat(table)
        self._involved.append(table)
        self.add(pre_stat)
        self.add('*')
    elif len(args) == 1 and isinstance(table := args[0], SqlCteExpr):
        self = SqlSelectStat(None)
        self.add(table)
        self.add(pre_stat)
        self.add('*')

    else:

        table, fields = select_from_fields(*args)
        if from_table is not None:
            table = from_table

        if table is None:
            raise RuntimeError('need to provide from_table')

        if isinstance(table, type):
            self = SqlSelectStat(None)
            self._involved.append(table)
        elif isinstance(table, SqlAlias) and isinstance(table._value, type):
            self = SqlSelectStat(None)
            self._involved.append(table)
        elif isinstance(table, SqlCteExpr):
            self = SqlSelectStat(None)
            self.add(table)
        else:
            self = SqlSelectStat(None)

        self.add(pre_stat)

        for i, field in enumerate(fields):
            if i > 0:
                self.add(',')

            if isinstance(field, SqlField):
                self.add(f'{field.table_name}.{field.name}')
            elif isinstance(field, SqlAlias) and isinstance(_field := field._value, SqlField):
                self.add([f'{_field.table_name}.{_field.name}', 'AS', repr(field._name)])
            elif isinstance(field, SqlAlias) and isinstance(expr := field._value, SqlExpr):
                self.add(expr)
                self.add(['AS', repr(field._name)])
            elif isinstance(field, SqlAliasField):
                self.add(field)
            elif isinstance(field, SqlFuncOper):
                self.add(field)
            elif isinstance(field, SqlLiteral):
                self.add(field)
            elif isinstance(field, SqlExpr):
                self.add(field)
            else:
                self.drop()
                raise TypeError('SELECT ' + repr(field))

    with self:
        self.add('FROM')
        if isinstance(table, str):
            self.add(table)
        elif isinstance(table, type):
            self.add(table_name(table))
        elif isinstance(table, SqlStat):
            self.add('(')
            self.add(table)
            self.add(')')
        elif isinstance(table, SqlAlias) and isinstance(_table := table._value, type):
            self.add([table_name(_table), table._name])
        elif isinstance(table, SqlAlias) and isinstance(subq := table._value, SqlSubQuery):
            self.add('(')
            self.add(subq.stat)
            self.add([')', 'AS', repr(table._name)])
        elif isinstance(table, SqlCteExpr):
            self.add(table._name)
        else:
            raise TypeError('FROM ' + repr(table))

    return self


def select_from_fields(*args) -> tuple[Union[type, SqlAlias, None], list[SqlExpr]]:
    if len(args) == 0:
        raise RuntimeError('empty field')

    table = None
    fields = []
    for arg in args:
        if isinstance(arg, (int, float, bool, str)):
            fields.append(SqlLiteral(repr(arg)))

        elif isinstance(arg, Field):
            if table is None:
                table = arg.table

            fields.append(SqlField(arg))

        elif isinstance(arg, SqlField):
            if table is None:
                table = arg.table

            fields.append(arg)

        elif isinstance(arg, SqlAlias) and isinstance(arg._value, SqlField):
            if table is None:
                table = arg._value.table

            fields.append(arg)

        elif isinstance(arg, SqlAliasField) and isinstance(field_table:=arg.table, type):
            if table is None:
                table = SqlAlias(field_table, arg.name)

            fields.append(arg)

        elif isinstance(arg, SqlAliasField) and isinstance(cte := arg.table, SqlCteExpr):
            if table is None:
                table = cte

            fields.append(arg)

        elif isinstance(arg, SqlExpr):
            if table is None:
                table =  use_table_first(arg)

            fields.append(arg)

        elif isinstance(arg, property):
            if (expr_table := getattr(arg.fget, '_sql_owner', None)) is None:
                raise RuntimeError(f'{arg} not a property from a Table')

            if table is None:
                table = expr_table

            if isinstance(expr := arg.fget(expr_table), SqlExpr):
                fields.append(expr)
            else:
                raise RuntimeError(f'{arg} does not return a sql expression.')
        else:
            raise TypeError(repr(arg))

    return table, fields


@overload
def insert_into(table: type[T], *, policy: UPDATE_POLICY = None, named=False) -> SqlInsertStat[T]:
    pass


@overload
def insert_into(*field, policy: UPDATE_POLICY = None, named=False) -> SqlInsertStat[T]:
    pass


def insert_into(*args, policy: UPDATE_POLICY = None, named=False) -> SqlInsertStat[T]:
    """
    ``INSERT``: https://www.sqlite.org/lang_insert.html

    insert values

    >>> insert_into(A, policy='REPLACE').build() # doctest: SKIP
    INSERT OR REPLACE INTO A VALUES (?)
    >>> insert_into(A, policy='REPLACE').submit([A(1), A(2)]) # doctest: SKIP

    insert values with field overwrite

    >>> insert_into(A, policy='REPLACE').values(a='1').build() # doctest: SKIP
    INSERT OR REPLACE INTO A VALUES (1)

    insert values from a table

    >>> insert_into(A, policy='IGNORE').select_from(B).build() # doctest: SKIP
    INSERT OR IGNORE INTO A
    SELECT * FROM B

    **features supporting**

    * `INSERT [OR ...]`
    * `VALUES`
    * `DEFAULT VALUES`
    * `SELECT`
    * upsert clause
    * returning clause

    **features not supporting**

    * `WITH [RECURSIVE]`

    :param table:
    :param policy:
    :param named:
    :return:
    """
    if len(args) == 1 and isinstance(args[0], type):
        self = SqlInsertStat((table := args[0]), named=named)
    else:
        table, fields = select_from_fields(*args)
        if table is None:
            raise RuntimeError()

        for i, field in enumerate(list(fields)):
            if isinstance(field, SqlField):
                if field.table != table:
                    raise RuntimeError(f'field {field.table_name}.{field.name} not belong to {table.__name__}')
                fields[i] = field.name
            else:
                raise TypeError()

        self = SqlInsertStat(table, fields, named=named)

    with self:
        self.add('INSERT')
        if policy is not None:
            self.add(['OR', policy.upper()])
        self.add(['INTO', table_name(table)])
        if self._fields is not None:
            self.add('(')
            for i, field in enumerate(self._fields):
                if i > 0:
                    self.add(',')
                self.add(field)
            self.add(')')
        return self


@overload
def replace_into(table: type[T], *, named=False) -> SqlInsertStat[T]:
    pass


@overload
def replace_into(*field: Any, named=False) -> SqlInsertStat[T]:
    pass


def replace_into(*args, named=False) -> SqlInsertStat[T]:
    return insert_into(*args, policy='REPLACE', named=named)


def update(table: type[T], *args: Union[bool, SqlCompareOper], **kwargs) -> SqlUpdateStat[T]:
    """
    ``UPDATE``: https://www.sqlite.org/lang_update.html

    >>> update(A, A.a==1).where(A.b==2).build() # doctest: SKIP
    UPDATE A SET A.a = 1 WHERE A.b = 2

     **features supporting**

    * `UPDATE [OR ...]`
    * `SET COLUMN = EXPR`
    * `FROM`
    * `WHERE`
    * `ON CONFLICT (COLUMNS) SET (COLUMNS) = EXPR`
    * returning clause

    **features not supporting**

    * `WITH [RECURSIVE]`
    * (qualified table name) `INDEXED BY`
    * (qualified table name) `NOT INDEXED`

    :param table:
    :param args:
    :param kwargs:
    :return:
    """
    with SqlUpdateStat(table) as self:
        self.add(['UPDATE', table_name(table), 'SET'])

        if len(args):
            for arg in args:
                self.add(SqlCompareOper.as_set_expr(arg))
                self.add(',')

        for term, value in kwargs.items():
            table_field(table, term)
            self.add(SqlLiteral(term) == value)
            self.add(',')
        self._stat.pop()

        return self


def delete_from(table: type[T]) -> SqlDeleteStat[T]:
    """
    ``DELETE``: https://www.sqlite.org/lang_delete.html

    >>> delete_from(A).where(A.b > 2).build()  # doctest: SKIP
    DELETE FROM A WHERE A.b > 2

    **features supporting**

    * `DELETE FROM`
    * `WHERE`
    * `ORDER BY`
    * `LIMIT [OFFSET]`
    * returning clause

    **features not supporting**

    * `WITH [RECURSIVE]`
    * (qualified table name) `INDEXED BY`
    * (qualified table name) `NOT INDEXED`

    :param table:
    :return:
    """
    with SqlDeleteStat(table) as self:
        self.add(['DELETE', 'FROM', table_name(table)])
        return self


def create_table(table: type[T], *, if_not_exists=True) -> SqlStat[T]:
    """
    ``CREATE``: https://www.sqlite.org/lang_createtable.html

    >>> @named_tuple_table_class # doctest: SKIP
    ... class A(NamedTuple):
    ...     a: int
    >>> create_table(A) # doctest: SKIP
    CREATE TABLE IF NOT EXISTS A (a INT NOT NULL)

    **features supporting**

    * `IF NOT EXISTS`
    * column constraint `NOT NULL`
    * column constraint `PRIMARY KEY`
    * column constraint `UNIQUE`
    * column constraint `CHECK`
    * column constraint `DEFAULT value`
    * table constraint `PRIMARY KEY`
    * table constraint `UNIQUE`
    * table constraint `CHECK`
    * table constraint `FOREIGN KEY`

    **features not supporting**

    * `CREATE TEMP|TEMPORARY`
    * `CREATE TEMP`
    * `AS SELECT`
    * column constraint `CONSTRAINT`
    * column constraint `NOT NULL ON CONFLICT`
    * column constraint `DEFAULT (EXPR)`
    * column constraint `COLLATE`
    * column constraint `REFERENCES`
    * column constraint `[GENERATED ALWAYS] AS`
    * table constraint `CONSTRAINT`
    * `WITHOUT ROWID`
    * `STRICT`

    :param table:
    :return:
    """
    with SqlStat(table) as self:
        self.add(['CREATE', 'TABLE'])
        if if_not_exists:
            self.add('IF NOT EXISTS')
        self.add(table_name(table))
        self.add('(')

        n_primary_key = len(primary_keys := table_primary_fields(table))

        for i, field in enumerate(table_fields(table)):
            if n_primary_key == 1 and field.is_primary:
                column_def(self, field, field.get_primary())
            else:
                column_def(self, field)
            self.add(',')

        if n_primary_key > 1:
            self.add(['PRIMARY KEY', '(', ' , '.join([it.name for it in primary_keys]), ')'])
            for it in primary_keys:
                if (conflict := it.get_primary().conflict) is not None:
                    self.add(['ON CONFLICT', conflict.upper()])
                    break
            self.add(',')

        for unique in table_unique_fields(table):
            if len(unique.fields) > 1:
                self.add(['UNIQUE', '(', ' , '.join(unique.fields), ')'])
                if (conflict := unique.conflict) is not None:
                    self.add(['ON CONFLICT', conflict.upper()])
                self.add(',')

        for foreign_key in table_foreign_fields(table):
            foreign_constraint(self, foreign_key)
            self.add(',')

        if (check := table_check_field(table, None)) is not None:
            check_constraint(self, check)
            self.add(',')

        self._stat.pop()
        self.add(')')

        return self


def column_def(self: SqlStat, field: Field, primary: PRIMARY = None):
    self.add(f"[{field.name}]")

    if field.sql_type == Any:
        pass
    elif field.sql_type == int:
        self.add('INTEGER')
    elif field.sql_type == float:
        self.add('FLOAT')
    elif field.sql_type == bool:
        self.add('BOOLEAN')
    elif field.sql_type == bytes:
        self.add('BLOB')
    elif field.sql_type == str:
        self.add('TEXT')
    elif field.sql_type == datetime.time:
        self.add('DATETIME')
    elif field.sql_type == datetime.date:
        self.add('DATETIME')
    elif field.sql_type == datetime.datetime:
        self.add('DATETIME')
    else:
        raise RuntimeError(f'field type {field.sql_type}')

    if field.not_null:
        if not field.has_default or field.f_value is not None:
            self.add('NOT NULL')

    if primary is not None:
        self.add(['PRIMARY', 'KEY'])
        if primary.order is not None:
            self.add(primary.order.upper())
        if primary.conflict is not None:
            self.add(['ON CONFLICT', primary.conflict.upper()])
        if primary.auto_increment:
            self.add('AUTOINCREMENT')

    elif (unique := field.get_unique()) is not None:
        self.add('UNIQUE')
        if unique.conflict is not None:
            self.add(['ON CONFLICT', unique.conflict.upper()])

    from .table import missing
    if field.f_value is missing:
        pass
    elif field.f_value is None:
        self.add('DEFAULT NULL')
    elif field.f_value == CURRENT_DATE:
        self.add('DEFAULT CURRENT_DATE')
    elif field.f_value == CURRENT_TIME:
        self.add('DEFAULT CURRENT_TIME')
    elif field.f_value == CURRENT_TIMESTAMP:
        self.add('DEFAULT CURRENT_TIMESTAMP')
    else:
        self.add(f'DEFAULT {repr(field.f_value)}')

    if (check := table_check_field(field.table, field.name)) is not None:
        check_constraint(self, check)


def foreign_constraint(self: SqlStat, foreign: ForeignConstraint):
    self.add(['FOREIGN KEY'])
    self.add('(')
    self.add(' , '.join(foreign.fields))
    self.add(')')
    self.add('REFERENCES')
    self.add(table_name(foreign.foreign_table))
    self.add('(')
    self.add(' , '.join(foreign.foreign_fields))
    self.add(')')
    if (policy := foreign.on_update) is not None:
        self.add(['ON UPDATE', policy])
    if (policy := foreign.on_delete) is not None:
        self.add(['ON DELETE', policy])


def check_constraint(self: SqlStat, check: CheckConstraint):
    self._deparameter = True
    self.add(['CHECK', '('])
    self.add(SqlRemovePlaceHolder(check.expression))
    self.add(')')
    self._deparameter = False
