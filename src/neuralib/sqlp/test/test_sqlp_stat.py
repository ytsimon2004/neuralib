import re
import unittest
from typing import NamedTuple, Annotated, Optional, Union

from neuralib import sqlp
from neuralib.sqlp import named_tuple_table_class, foreign, PRIMARY, check


@sqlp.named_tuple_table_class
class Test(NamedTuple):
    a: Annotated[str, sqlp.PRIMARY]
    b: int


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

        stat = sqlp.create_table(Test).build()
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test ( 
            a TEXT NOT NULL , 
            b INT NOT NULL 
        )
        """, stat)

    def test_create_table_with_primary_key(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: Annotated[str, sqlp.PRIMARY]
            b: int

        stat = sqlp.create_table(Test).build()
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test ( 
            a TEXT NOT NULL , 
            b INT NOT NULL ,
            PRIMARY KEY ( a )
        )
        """, stat)

    def test_create_table_with_unique_key(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: Annotated[str, sqlp.UNIQUE]
            b: int

        stat = sqlp.create_table(Test).build()
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test ( 
            a TEXT NOT NULL , 
            b INT NOT NULL ,
            UNIQUE ( a )
        )
        """, stat)

    def test_create_table_with_null(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: Optional[str]
            b: Union[int, None]
            c: Union[None, float]

        stat = sqlp.create_table(Test).build()
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test ( 
            a TEXT , 
            b INT ,
            c FLOAT
        )
        """, stat)

    def test_create_table_foreign(self):
        @named_tuple_table_class
        class Ref(NamedTuple):
            name: Annotated[str, PRIMARY]

        @named_tuple_table_class
        class Test(NamedTuple):
            name: Annotated[str, PRIMARY]
            ref: str

            @foreign(Ref.name, update='NO ACTION', delete='NO ACTION')
            def _ref(self):
                return self.ref

        stat = sqlp.create_table(Test).build()
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test ( 
            name TEXT NOT NULL ,
            ref TEXT NOT NULL , 
            PRIMARY KEY ( name ) ,
            FOREIGN KEY ( ref ) REFERENCES Ref ( name )
                ON UPDATE NO ACTION ON DELETE NO ACTION
        )
        """, stat)

    def test_create_table_foreign_self(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            name: Annotated[str, PRIMARY]
            ref: str

            @foreign('name', update='NO ACTION', delete='NO ACTION')
            def _ref(self):
                return self.ref

        stat = sqlp.create_table(Test).build()
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test ( 
            name TEXT NOT NULL ,
            ref TEXT NOT NULL , 
            PRIMARY KEY ( name ) ,
            FOREIGN KEY ( ref ) REFERENCES Test ( name )
                ON UPDATE NO ACTION ON DELETE NO ACTION
        )
        """, stat)

    def test_create_table_check_field(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            name: Annotated[str, PRIMARY]
            age: int

            @check('age')
            def _age(self) -> bool:
                return self.age > 10

        stat = sqlp.create_table(Test).build()
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test ( 
            name TEXT NOT NULL ,
            age INT NOT NULL CHECK ( Test.age > 10 ) , 
            PRIMARY KEY ( name ) 
        )
        """, stat)

    def test_create_table_check(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            name: Annotated[str, PRIMARY]
            age: int

            @check()
            def _check_all(self) -> bool:
                return (self.age > 10) & (self.name != '')

        stat = sqlp.create_table(Test).build()
        print(stat)
        self.assert_sql_state_equal("""
        CREATE TABLE IF NOT EXISTS Test ( 
            name TEXT NOT NULL ,
            age INT NOT NULL , 
            PRIMARY KEY ( name ) ,
            CHECK ( Test.age > 10 AND Test.name != '' )
        )
        """, stat)

    def test_select_from(self):
        stat = sqlp.select_from(Test).build()
        self.assert_sql_state_equal("""
        SELECT * FROM Test
        """, stat)

    def test_select_from_fields(self):
        stat = sqlp.select_from(Test.a).build()
        self.assert_sql_state_equal("""
        SELECT Test.a FROM Test
        """, stat)

        stat = sqlp.select_from(Test.a, Test.b).build()
        self.assert_sql_state_equal("""
        SELECT Test.a , Test.b FROM Test
        """, stat)

    def test_select_from_where(self):
        stat = sqlp.select_from(Test).where(
            Test.a == 1
        )

        self.assert_sql_state_equal("""
        SELECT * FROM Test
        WHERE Test.a = ? 
        """, stat.build())
        self.assertListEqual([1], stat._para)

    def test_select_from_where_and(self):
        stat = sqlp.select_from(Test).where(
            Test.a == '1',
            Test.b == 0
        )
        self.assert_sql_state_equal("""
        SELECT * FROM Test
        WHERE Test.a = ? AND Test.b = ? 
        """, stat.build())
        self.assertListEqual(['1', 0], stat._para)

    def test_select_from_where_and_oper(self):
        stat = sqlp.select_from(Test).where(
            (Test.a == '1') & (Test.b == 0)
        )

        self.assert_sql_state_equal("""
        SELECT * FROM Test
        WHERE Test.a = ? AND Test.b = ?
        """, stat.build())
        self.assertListEqual(['1', 0], stat._para)

    def test_select_from_where_or_oper(self):
        stat = sqlp.select_from(Test).where(
            (Test.a == '1') | (Test.b == 0)
        )

        self.assert_sql_state_equal("""
        SELECT * FROM Test
        WHERE Test.a = ? OR Test.b = ?
        """, stat.build())
        self.assertListEqual(['1', 0], stat._para)

    def test_select_from_limit(self):
        stat = sqlp.select_from(Test).limit(10)

        self.assert_sql_state_equal("""
        SELECT * FROM Test
        LIMIT 10
        """, stat.build())

    def test_select_from_limit_offset(self):
        stat = sqlp.select_from(Test).limit(10, 2)

        self.assert_sql_state_equal("""
        SELECT * FROM Test
        LIMIT 10 OFFSET 2
        """, stat.build())

    def test_select_from_order_by(self):
        stat = sqlp.select_from(Test).order_by(Test.a)

        self.assert_sql_state_equal("""
        SELECT * FROM Test
        ORDER BY Test.a
        """, stat.build())

    def test_select_from_order_by_asc(self):
        stat = sqlp.select_from(Test).order_by(
            sqlp.asc(Test.a), sqlp.desc(Test.b)
        )

        self.assert_sql_state_equal("""
        SELECT * FROM Test
        ORDER BY Test.a ASC , Test.b DESC
        """, stat.build())

    def test_insert_into(self):
        stat = sqlp.insert_into(Test).build()

        self.assert_sql_state_equal("""
        INSERT INTO Test VALUES ( ? , ? )
        """, stat)

    def test_insert_into_or_replace(self):
        stat = sqlp.insert_into(Test, policy='REPLACE').build()

        self.assert_sql_state_equal("""
        INSERT OR REPLACE INTO Test VALUES ( ? , ? )
        """, stat)

    def test_insert_into_named(self):
        stat = sqlp.insert_into(Test, named=True).build()

        self.assert_sql_state_equal("""
        INSERT INTO Test VALUES ( :a , :b )
        """, stat)

    def test_insert_into_named_overwrite(self):
        stat = sqlp.insert_into(Test, named=True).values(
            b=sqlp.max(10, ':b')
        ).build()

        self.assert_sql_state_equal("""
        INSERT INTO Test VALUES ( :a , MAX ( 10 , :b ) )
        """, stat)

    def test_insert_into_from_select(self):
        @sqlp.named_tuple_table_class
        class Other(NamedTuple):
            a: Annotated[str, sqlp.PRIMARY]
            b: int

        stat = sqlp.insert_into(Test).select_from(Other).build()
        self.assert_sql_state_equal("""
        INSERT INTO Test ( a , b ) 
        SELECT * FROM Other
        """, stat)

    def test_update_key_value(self):
        stat = sqlp.update(Test, b=1)

        self.assert_sql_state_equal("""
        UPDATE Test SET
        b = ?
        """, stat.build())
        self.assertListEqual([1], stat._para)

    def test_update_field(self):
        stat = sqlp.update(Test, Test.b == 1)

        self.assert_sql_state_equal("""
        UPDATE Test SET
        b = ?
        """, stat.build())
        self.assertListEqual([1], stat._para)

    def test_update_field_where(self):
        stat = sqlp.update(Test, Test.b == 1).where(Test.a == '1')

        self.assert_sql_state_equal("""
        UPDATE Test SET
            b = ?
        WHERE Test.a = ?
        """, stat.build())
        self.assertListEqual([1, '1'], stat._para)

    # TODO sqlp.delete_from

    def test_alias(self):
        stat = sqlp.select_from(
            Test.a @ 'x',
            Test.b @ 'y',
        )
        self.assert_sql_state_equal("""
        SELECT
            Test.a AS 'x' ,
            Test.b AS 'y'
        FROM Test
        """, stat.build())

    def test_over_window_func_over(self):
        stat = sqlp.select_from(
            Test.a,
            Test.b,
            sqlp.row_number().over(order_by=Test.b)
        )
        self.assert_sql_state_equal("""
        SELECT
            Test.a , Test.b ,
            ROW_NUMBER () OVER ( ORDER BY Test.b )
        FROM Test
        """, stat.build())

    def test_over_window_func_partition_by(self):
        stat = sqlp.select_from(
            Test.a,
            Test.b,
            sqlp.row_number().over(partition_by=Test.b)
        )

        self.assert_sql_state_equal("""
        SELECT
            Test.a , Test.b ,
            ROW_NUMBER () OVER ( PARTITION BY Test.b )
        FROM Test
        """, stat.build())

    def test_over_window_func_frame(self):
        w = sqlp.window_def('w').over(order_by=Test.a)
        with w.frame('RANGE') as f:
            f.between(f.unbounded_preceding(), f.current_row())

        stat = sqlp.select_from(
            Test.a,
            Test.b,
            sqlp.row_number().over(w)
        )

        self.assert_sql_state_equal("""
        SELECT
            Test.a , Test.b ,
            ROW_NUMBER () OVER w
        FROM Test
        WINDOW w AS ( ORDER BY Test.a RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW )
        """, stat.build())

    def test_join(self):
        @sqlp.named_tuple_table_class
        class Other(NamedTuple):
            c: Annotated[str, sqlp.PRIMARY]
            d: int

        stat = sqlp.select_from(
            Test.a, Other.c
        ).join(Other, by='inner').on(Test.b == Other.d)

        self.assert_sql_state_equal("""
        SELECT
            Test.a , Other.c
        FROM Test
        INNER JOIN Other ON Test.b = Other.d 
        """, stat.build())


if __name__ == '__main__':
    unittest.main()
