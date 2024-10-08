import datetime
import re
import unittest
from typing import NamedTuple, Annotated, Optional, Union

from neuralib import sqlp
from neuralib.sqlp import named_tuple_table_class, foreign, check


@sqlp.named_tuple_table_class
class Test(NamedTuple):
    a: Annotated[str, sqlp.PRIMARY]
    b: int
    c: int


class SqlpStatTest(unittest.TestCase):
    def assert_sql_state_equal(self, a: str, b: str):
        a = re.split(' +', a.replace('\n', ' ').strip())
        b = re.split(' +', b.replace('\n', ' ').strip())
        self.assertListEqual(a, b)

    def test_create_table(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: str
            b: int

        stat, par = sqlp.create_table(Test).build()
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test (
            [a] TEXT NOT NULL ,
            [b] INTEGER NOT NULL
        )
        """, stat)
        self.assertListEqual([], par)

    def test_create_table_with_primary_key(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: Annotated[str, sqlp.PRIMARY]
            b: int

        stat, par = sqlp.create_table(Test).build()
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test (
            [a] TEXT NOT NULL PRIMARY KEY ,
            [b] INTEGER NOT NULL
        )
        """, stat)
        self.assertListEqual([], par)

    def test_create_table_with_primary_keys(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: Annotated[str, sqlp.PRIMARY]
            b: Annotated[int, sqlp.PRIMARY]

        stat, par = sqlp.create_table(Test).build()
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test (
            [a] TEXT NOT NULL ,
            [b] INTEGER NOT NULL ,
            PRIMARY KEY ( a , b )
        )
        """, stat)
        self.assertListEqual([], par)

    def test_create_table_with_unique_key(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: Annotated[str, sqlp.UNIQUE]
            b: int

        stat, par = sqlp.create_table(Test).build()
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test (
            [a] TEXT NOT NULL UNIQUE ,
            [b] INTEGER NOT NULL
        )
        """, stat)
        self.assertListEqual([], par)

    def test_create_table_with_unique_keys(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: Annotated[str, sqlp.UNIQUE]
            b: Annotated[str, sqlp.UNIQUE]

        stat, par = sqlp.create_table(Test).build()
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test (
            [a] TEXT NOT NULL UNIQUE ,
            [b] TEXT NOT NULL UNIQUE
        )
        """, stat)
        self.assertListEqual([], par)

    def test_create_table_with_unique_keys_on_table(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: str
            b: str

            @sqlp.unique()
            def _unique_ab_pair(self):
                return self.a, self.b

        stat, par = sqlp.create_table(Test).build()
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test (
            [a] TEXT NOT NULL ,
            [b] TEXT NOT NULL ,
            UNIQUE ( a , b )
        )
        """, stat)
        self.assertListEqual([], par)

    def test_create_table_with_null(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: Optional[str]
            b: Union[int, None]
            c: Union[None, float]

        stat, par = sqlp.create_table(Test).build()
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test (
            [a] TEXT ,
            [b] INTEGER ,
            [c] FLOAT
        )
        """, stat)
        self.assertListEqual([], par)

    def test_create_table_with_auto_increment(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: Annotated[int, sqlp.PRIMARY(auto_increment=True)]

        stat, par = sqlp.create_table(Test).build()
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test (
            [a] INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT
        )
        """, stat)
        self.assertListEqual([], par)

    def test_create_table_with_default_date_time(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: Annotated[datetime.date, sqlp.CURRENT_DATE]
            b: Annotated[datetime.time, sqlp.CURRENT_TIME]
            c: Annotated[datetime.datetime, sqlp.CURRENT_TIMESTAMP]

        stat, par = sqlp.create_table(Test).build()
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test (
            [a] DATETIME NOT NULL DEFAULT CURRENT_DATE ,
            [b] DATETIME NOT NULL DEFAULT CURRENT_TIME ,
            [c] DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """, stat)
        self.assertListEqual([], par)

    def test_create_table_foreign(self):
        @named_tuple_table_class
        class Ref(NamedTuple):
            name: Annotated[str, sqlp.PRIMARY]
            other: str

        @named_tuple_table_class
        class Test(NamedTuple):
            name: Annotated[str, sqlp.PRIMARY]
            ref: str

            @foreign(Ref.name, update='NO ACTION', delete='NO ACTION')
            def _ref(self):
                return self.ref

        stat, par = sqlp.create_table(Test).build()
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test (
            [name] TEXT NOT NULL PRIMARY KEY ,
            [ref] TEXT NOT NULL ,
            FOREIGN KEY ( ref ) REFERENCES Ref ( name )
                ON UPDATE NO ACTION ON DELETE NO ACTION
        )
        """, stat)
        self.assertListEqual([], par)

    def test_create_table_foreign_primary(self):
        @named_tuple_table_class
        class Ref(NamedTuple):
            name: Annotated[str, sqlp.PRIMARY]
            other: str

        @named_tuple_table_class
        class Test(NamedTuple):
            name: Annotated[str, sqlp.PRIMARY]
            ref: str

            @foreign(Ref, update='NO ACTION', delete='NO ACTION')
            def _ref(self):
                return self.ref

        stat, par = sqlp.create_table(Test).build()
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test (
            [name] TEXT NOT NULL PRIMARY KEY ,
            [ref] TEXT NOT NULL ,
            FOREIGN KEY ( ref ) REFERENCES Ref ( name )
                ON UPDATE NO ACTION ON DELETE NO ACTION
        )
        """, stat)
        self.assertListEqual([], par)

    def test_create_table_foreign_self(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            name: Annotated[str, sqlp.PRIMARY]
            ref: str

            @foreign('name', update='NO ACTION', delete='NO ACTION')
            def _ref(self):
                return self.ref

        stat, par = sqlp.create_table(Test).build()
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test (
            [name] TEXT NOT NULL PRIMARY KEY ,
            [ref] TEXT NOT NULL ,
            FOREIGN KEY ( ref ) REFERENCES Test ( name )
                ON UPDATE NO ACTION ON DELETE NO ACTION
        )
        """, stat)
        self.assertListEqual([], par)

    def test_create_table_check_field(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            name: Annotated[str, sqlp.PRIMARY]
            age: int

            @check('age')
            def _age(self) -> bool:
                return self.age > 10

        stat, par = sqlp.create_table(Test).build()
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test (
            [name] TEXT NOT NULL PRIMARY KEY ,
            [age] INTEGER NOT NULL CHECK ( Test.age > 10 )
        )
        """, stat)
        self.assertListEqual([], par)

    def test_create_table_check(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            name: Annotated[str, sqlp.PRIMARY]
            age: int

            @check()
            def _check_all(self) -> bool:
                return (self.age > 10) & (self.name != '')

        stat, par = sqlp.create_table(Test).build()
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test (
            [name] TEXT NOT NULL PRIMARY KEY ,
            [age] INTEGER NOT NULL ,
            CHECK ( ( Test.age > 10 AND Test.name != '' ) )
        )
        """, stat)
        self.assertListEqual([], par)

    def test_select_from(self):
        stat, par = sqlp.select_from(Test).build()
        self.assert_sql_state_equal("""
        SELECT * FROM Test
        """, stat)
        self.assertListEqual([], par)

    def test_select_from_fields(self):
        stat, par = sqlp.select_from(Test.a).build()
        self.assert_sql_state_equal("""
        SELECT Test.a FROM Test
        """, stat)
        self.assertListEqual([], par)

        stat, par = sqlp.select_from(Test.a, Test.b).build()
        self.assert_sql_state_equal("""
        SELECT Test.a , Test.b FROM Test
        """, stat)
        self.assertListEqual([], par)

    def test_select_from_where(self):
        stat = sqlp.select_from(Test).where(
            Test.a == 1
        )

        res, par = stat.build()
        self.assert_sql_state_equal("""
        SELECT * FROM Test
        WHERE Test.a = 1
        """, res)
        self.assertListEqual([], par)

    def test_select_from_where_and(self):
        stat = sqlp.select_from(Test).where(
            Test.a == '1',
            Test.b == 0
        )

        res, par = stat.build()
        self.assert_sql_state_equal("""
        SELECT * FROM Test
        WHERE ( Test.a = ? AND Test.b = 0 )
        """, res)
        self.assertListEqual(['1'], par)

    def test_select_from_where_and_oper(self):
        stat = sqlp.select_from(Test).where(
            (Test.a == '1') & (Test.b == 0)
        )

        res, par = stat.build()
        self.assert_sql_state_equal("""
        SELECT * FROM Test
        WHERE ( Test.a = ? AND Test.b = 0 )
        """, res)
        self.assertListEqual(['1'], par)

    def test_select_from_where_or_oper(self):
        stat = sqlp.select_from(Test).where(
            (Test.a == '1') | (Test.b == 0)
        )

        res, par = stat.build()
        self.assert_sql_state_equal("""
        SELECT * FROM Test
        WHERE ( Test.a = ? OR Test.b = 0 )
        """, res)
        self.assertListEqual(['1'], par)

    def test_select_from_limit(self):
        stat = sqlp.select_from(Test).limit(10)

        res, par = stat.build()
        self.assert_sql_state_equal("""
        SELECT * FROM Test
        LIMIT 10
        """, res)
        self.assertListEqual([], par)

    def test_select_from_limit_offset(self):
        stat = sqlp.select_from(Test).limit(10, 2)

        res, par = stat.build()
        self.assert_sql_state_equal("""
        SELECT * FROM Test
        LIMIT 10 OFFSET 2
        """, res)
        self.assertListEqual([], par)

    def test_select_from_order_by(self):
        stat = sqlp.select_from(Test).order_by(Test.a)

        res, par = stat.build()
        self.assert_sql_state_equal("""
        SELECT * FROM Test
        ORDER BY Test.a
        """, res)
        self.assertListEqual([], par)

    def test_select_from_order_by_asc(self):
        stat = sqlp.select_from(Test).order_by(
            sqlp.asc(Test.a), sqlp.desc(Test.b)
        )

        res, par = stat.build()
        self.assert_sql_state_equal("""
        SELECT * FROM Test
        ORDER BY Test.a ASC , Test.b DESC
        """, res)
        self.assertListEqual([], par)

    def test_insert_into(self):
        stat, par = sqlp.insert_into(Test).build()

        self.assert_sql_state_equal("""
        INSERT INTO Test VALUES ( ? , ? , ? )
        """, stat)
        self.assertListEqual([], par)

    def test_insert_into_or_replace(self):
        stat, par = sqlp.insert_into(Test, policy='REPLACE').build()

        self.assert_sql_state_equal("""
        INSERT OR REPLACE INTO Test VALUES ( ? , ? , ? )
        """, stat)
        self.assertListEqual([], par)

    def test_insert_into_named(self):
        stat, par = sqlp.insert_into(Test, named=True).build()

        self.assert_sql_state_equal("""
        INSERT INTO Test VALUES ( :a , :b , :c )
        """, stat)
        self.assertListEqual([], par)

    def test_insert_into_named_overwrite_keyword(self):
        stat, par = sqlp.insert_into(Test, named=True).values(
            b=sqlp.max(10, ':b')
        ).build()

        self.assert_sql_state_equal("""
        INSERT INTO Test VALUES ( :a , MAX ( 10 , :b ) , :c )
        """, stat)
        self.assertListEqual([], par)

    def test_insert_into_partial(self):
        stat, par = sqlp.insert_into(Test.b).build()

        self.assert_sql_state_equal("""
        INSERT INTO Test ( b ) VALUES ( ? )
        """, stat)
        self.assertListEqual([], par)

    def test_insert_into_constant_value(self):
        stat, par = sqlp.insert_into(Test.a, Test.b).values(
            Test.b == 1
        ).build()

        self.assert_sql_state_equal("""
        INSERT INTO Test ( a , b ) VALUES ( ? , 1 )
        """, stat)
        self.assertListEqual([], par)

    def test_insert_into_from_select(self):
        @sqlp.named_tuple_table_class
        class Other(NamedTuple):
            a: Annotated[str, sqlp.PRIMARY]
            b: int

        stat, par = sqlp.insert_into(Test).select_from(Other).build()
        self.assert_sql_state_equal("""
        INSERT INTO Test ( a , b , c )
        SELECT * FROM Other
        """, stat)
        self.assertListEqual([], par)

    def test_update_key_value(self):
        stat = sqlp.update(Test, b=1, c=2)

        res, par = stat.build()
        self.assert_sql_state_equal("""
        UPDATE Test SET
        b = 1 ,
        c = 2
        """, res)
        self.assertListEqual([], par)

    def test_update_field(self):
        stat = sqlp.update(Test, Test.b == 1, Test.c == 3)

        res, par = stat.build()
        self.assert_sql_state_equal("""
        UPDATE Test SET
        b = 1 ,
        c = 3
        """, res)
        self.assertListEqual([], par)

    def test_update_field_where(self):
        stat = sqlp.update(Test, Test.b == 1).where(Test.a == '1')

        res, par = stat.build()
        self.assert_sql_state_equal("""
        UPDATE Test SET
            b = 1
        WHERE Test.a = ?
        """, res)
        self.assertListEqual(['1'], par)

    # TODO sqlp.delete_from

    def test_alias(self):
        stat = sqlp.select_from(
            Test.a @ 'x',
            Test.b @ 'y',
        )

        res, par = stat.build()
        self.assert_sql_state_equal("""
        SELECT
            Test.a AS 'x' ,
            Test.b AS 'y'
        FROM Test
        """, res)
        self.assertListEqual([], par)

    def test_table_alias(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            name: Annotated[str, sqlp.PRIMARY]
            age: int

        t = sqlp.alias(Test, 't')

        stat = sqlp.select_from(t.name)
        res, par = stat.build()
        self.assert_sql_state_equal("""
        SELECT t.name FROM Test t
        """, res)
        self.assertListEqual([], par)

    def test_over_window_func_over(self):
        stat = sqlp.select_from(
            Test.a,
            Test.b,
            sqlp.row_number().over(order_by=Test.b)
        )
        res, par = stat.build()
        self.assert_sql_state_equal("""
        SELECT
            Test.a , Test.b ,
            ROW_NUMBER ( ) OVER ( ORDER BY Test.b )
        FROM Test
        """, res)
        self.assertListEqual([], par)

    def test_over_window_func_partition_by(self):
        stat = sqlp.select_from(
            Test.a,
            Test.b,
            sqlp.row_number().over(partition_by=Test.b)
        )

        res, par = stat.build()
        self.assert_sql_state_equal("""
        SELECT
            Test.a , Test.b ,
            ROW_NUMBER ( ) OVER ( PARTITION BY Test.b )
        FROM Test
        """, res)
        self.assertListEqual([], par)

    def test_over_window_func_frame(self):
        w = sqlp.window_def('w').over(order_by=Test.a)
        with w.frame('RANGE') as f:
            f.between(f.unbounded_preceding(), f.current_row())

        stat = sqlp.select_from(
            Test.a,
            Test.b,
            sqlp.row_number().over(w)
        ).windows(w)

        res, par = stat.build()
        self.assert_sql_state_equal("""
        SELECT
            Test.a , Test.b ,
            ROW_NUMBER ( ) OVER w
        FROM Test
        WINDOW w AS ( ORDER BY Test.a RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW )
        """, res)
        self.assertListEqual([], par)

    def test_join(self):
        @sqlp.named_tuple_table_class
        class Other(NamedTuple):
            c: Annotated[str, sqlp.PRIMARY]
            d: int

        stat = sqlp.select_from(
            Test.a, Other.c
        ).join(Test.b == Other.d, by='inner')

        res, par = stat.build()
        self.assert_sql_state_equal("""
        SELECT
            Test.a , Other.c
        FROM Test
        INNER JOIN Other ON Test.b = Other.d
        """, res)
        self.assertListEqual([], par)


if __name__ == '__main__':
    unittest.main()
