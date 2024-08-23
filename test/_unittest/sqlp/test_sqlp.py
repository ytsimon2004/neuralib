import datetime
import re
import sqlite3
import unittest
from pathlib import Path
from typing import NamedTuple, Annotated, Optional, Union

from neuralib import sqlp
from neuralib.sqlp import named_tuple_table_class, PRIMARY, Connection, check, foreign
from neuralib.sqlp.stat import SqlStat


@named_tuple_table_class
class Person(NamedTuple):
    name: Annotated[str, PRIMARY]
    age: int

    @check('age')
    def _age(self):
        return self.age > 10


@named_tuple_table_class
class Bank(NamedTuple):
    name: Annotated[str, PRIMARY]


@named_tuple_table_class
class Account(NamedTuple):
    bank: Annotated[str, PRIMARY]
    person: Annotated[str, PRIMARY]
    money: int

    @foreign(Person)
    def _person(self):
        return self.person

    @foreign(Bank)
    def _bank(self):
        return self.bank


class SqlpTableTest(unittest.TestCase):
    conn: Connection

    @classmethod
    def setUpClass(cls):
        cls.conn = Connection(debug=True)
        with cls.conn:
            sqlp.create_table(Person).submit()
            sqlp.create_table(Bank).submit()
            sqlp.create_table(Account).submit()

            sqlp.insert_into(Person).submit([
                Person('Alice', 18),
                Person('Bob', 20),
            ])

            sqlp.insert_into(Bank).submit([
                Bank('V'),
                Bank('M'),
            ])

            sqlp.insert_into(Account).submit([
                Account('V', 'Alice', 1000),
                Account('V', 'Bob', 2000),
                Account('M', 'Alice', 200),
                Account('M', 'Bob', 100),
            ])

    def setUp(self):
        self.conn.__enter__()

    def tearDown(self):
        self.conn.__exit__(None, None, None)

    def assertSqlExeEqual(self, raw_sql: str, stat: SqlStat, parameter=()):
        r1 = self.conn.execute(raw_sql, parameter).fetchall()
        r2 = self.conn.execute(stat, parameter).fetchall()
        print(stat.build())
        self.assertListEqual(r1, r2)

    def test_insert_fail_on_check(self):
        with self.assertRaises(sqlite3.IntegrityError):
            sqlp.insert_into(Person).submit([Person('baby', 1)])

    def test_select_person(self):
        results = sqlp.select_from(Person).order_by(Person.name).fetchall()

        self.assertListEqual([
            Person('Alice', 18),
            Person('Bob', 20),
        ], results)

    def test_select_with_literal(self):
        results = sqlp.select_from(Person.name, Person.age, 1).order_by(Person.name).fetchall()

        self.assertListEqual([
            ('Alice', 18, 1),
            ('Bob', 20, 1),
        ], results)

    def test_where(self):
        results = sqlp.select_from(Account).where(
            Account.bank == 'V'
        ).fetchall()

        self.assertListEqual([
            Account('V', 'Alice', 1000),
            Account('V', 'Bob', 2000),
        ], results)

    def test_update(self):
        sqlp.insert_into(Account, policy='REPLACE').submit([Account('K', 'Eve', 0)])
        self.assertEqual(Account('K', 'Eve', 0), sqlp.select_from(Account).where(Account.bank == 'K').fetchone())
        sqlp.update(Account, Account.money == 100).where(
            Account.bank == 'K'
        ).submit()
        self.assertEqual(Account('K', 'Eve', 100), sqlp.select_from(Account).where(Account.bank == 'K').fetchone())

    def test_delete(self):
        sqlp.insert_into(Account, policy='REPLACE').submit([Account('K', 'Eve', 0)])
        self.assertEqual(Account('K', 'Eve', 0), sqlp.select_from(Account).where(Account.bank == 'K').fetchone())
        sqlp.delete_from(Account).where(Account.bank == 'K').submit()
        self.assertIsNone(sqlp.select_from(Account).where(Account.bank == 'K').fetchone())

    def test_foreign_field(self):
        from neuralib.sqlp.table import table_foreign_field, ForeignConstraint
        person = table_foreign_field(Account, Person)
        self.assertIsInstance(person, ForeignConstraint)
        self.assertEqual(person.name, '_person')
        self.assertEqual(person.foreign_table, Person)
        self.assertListEqual(person.foreign_fields, ['name'])
        self.assertEqual(person.table, Account)
        self.assertListEqual(person.fields, ['person'])

    def test_foreign_field_callable(self):
        from neuralib.sqlp.table import table_foreign_field, ForeignConstraint
        person = table_foreign_field(Account._person)
        self.assertIsInstance(person, ForeignConstraint)
        self.assertEqual(person.name, '_person')
        self.assertEqual(person.foreign_table, Person)
        self.assertListEqual(person.foreign_fields, ['name'])
        self.assertEqual(person.table, Account)
        self.assertListEqual(person.fields, ['person'])

    def test_map_foreign(self):
        from neuralib.sqlp.util import map_foreign
        results = map_foreign(Account('V', 'Alice', 1000), Person).fetchall()
        self.assertListEqual([
            Person('Alice', 18),
        ], results)

    def test_pull_foreign(self):
        from neuralib.sqlp.util import pull_foreign
        results = pull_foreign(Account, Person('Alice', 18)).fetchall()
        self.assertListEqual([
            Account('V', 'Alice', 1000),
            Account('M', 'Alice', 200),
        ], results)


class SqlCreateTableTest(unittest.TestCase):
    conn: Connection

    def setUp(self):
        self.conn = Connection(debug=True)
        self.conn.__enter__()

    def tearDown(self):
        self.conn.__exit__(None, None, None)

    def test_create_table(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: str
            b: int

        sqlp.create_table(Test)

    def test_create_table_with_primary_key(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: Annotated[str, sqlp.PRIMARY]
            b: int

        sqlp.create_table(Test)

    def test_create_table_with_primary_keys(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: Annotated[str, sqlp.PRIMARY]
            b: Annotated[int, sqlp.PRIMARY]

        sqlp.create_table(Test)

    def test_create_table_with_unique_key(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: Annotated[str, sqlp.UNIQUE]
            b: int

        sqlp.create_table(Test)

    def test_create_table_with_unique_keys(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: Annotated[str, sqlp.UNIQUE]
            b: Annotated[str, sqlp.UNIQUE]

        sqlp.create_table(Test)

    def test_create_table_with_unique_keys_on_table(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: str
            b: str

            @sqlp.unique()
            def _unique_ab_pair(self):
                return self.a, self.b

        sqlp.create_table(Test)

    def test_create_table_with_null(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: Optional[str]
            b: Union[int, None]
            c: Union[None, float]

        sqlp.create_table(Test)

    def test_create_table_with_auto_increment(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: Annotated[int, sqlp.PRIMARY(auto_increment=True)]

        sqlp.create_table(Test)

    def test_create_table_with_default_date_time(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: Annotated[datetime.date, sqlp.CURRENT_DATE]
            b: Annotated[datetime.time, sqlp.CURRENT_TIME]
            c: Annotated[datetime.datetime, sqlp.CURRENT_TIMESTAMP]

        sqlp.create_table(Test)

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

        sqlp.create_table(Ref)
        sqlp.create_table(Test)

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

        sqlp.create_table(Ref)
        sqlp.create_table(Test)

    def test_create_table_foreign_self(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            name: Annotated[str, sqlp.PRIMARY]
            ref: str

            @foreign('name', update='NO ACTION', delete='NO ACTION')
            def _ref(self):
                return self.ref

        sqlp.create_table(Test)

    def test_create_table_check_field(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            name: Annotated[str, sqlp.PRIMARY]
            age: int

            @check('age')
            def _age(self) -> bool:
                return self.age > 10

        sqlp.create_table(Test)

    def test_create_table_check(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            name: Annotated[str, sqlp.PRIMARY]
            age: int

            @check()
            def _check_all(self) -> bool:
                return (self.age > 10) & (self.name != '')

        sqlp.create_table(Test)

    def test_fail_on_stat_constructing(self):
        @named_tuple_table_class
        class Test(NamedTuple):
            a: str
            b: str

        sqlp.create_table(Test)
        sqlp.insert_into(Test).submit([Test('a', '1'), Test('b', '2')])

        with self.assertRaises(TypeError) as capture:
            sqlp.delete_from(Test).where(Test.a == object()).build()

        print(capture.exception)

        self.assertListEqual([Test('a', '1'), Test('b', '2')],
                             sqlp.select_from(Test).fetchall())


class SqlUpsertTest(unittest.TestCase):
    conn: Connection

    def setUp(self):
        self.conn = Connection(debug=True)
        with self.conn:
            sqlp.create_table(Person).submit()
            sqlp.create_table(Bank).submit()
            sqlp.create_table(Account).submit()

            sqlp.insert_into(Person).submit([
                Person('Alice', 18),
                Person('Bob', 20),
            ])

            sqlp.insert_into(Bank).submit([
                Bank('V'),
                Bank('M'),
            ])

            sqlp.insert_into(Account).submit([
                Account('V', 'Alice', 1000),
                Account('V', 'Bob', 2000),
                Account('M', 'Alice', 200),
                Account('M', 'Bob', 100),
            ])

    def test_update_conflict_with_name(self):
        with self.conn:
            sqlp.insert_into(Person).on_conflict(Person.name).do_update(
                Person.age == sqlp.excluded(Person).age
            ).submit(
                [Person('Alice', 28)]
            )

            result = sqlp.select_from(Person).where(Person.name == 'Alice').fetchone()
            self.assertEqual(result, Person('Alice', 28))

    def test_update_conflict(self):
        with self.conn:
            sqlp.insert_into(Person).on_conflict().do_nothing().submit(
                [Person('Alice', 28)]
            )

            result = sqlp.select_from(Person).where(Person.name == 'Alice').fetchone()
            self.assertEqual(result, Person('Alice', 18))


class SqlTableOtherTest(unittest.TestCase):
    def assert_sql_state_equal(self, a: str, b: str):
        a = re.split(' +', a.replace('\n', ' ').strip())
        b = re.split(' +', b.replace('\n', ' ').strip())
        self.assertListEqual(a, b)

    def test_create_table_not_null(self):
        @named_tuple_table_class
        class A(NamedTuple):
            a: int
            b: Optional[int]

        with Connection(debug=True) as conn:
            sqlp.create_table(A)
            self.assert_sql_state_equal("""
            CREATE TABLE A (
                [a] INTEGER NOT NULL ,
                [b] INTEGER
            )
            """, conn.table_schema(A))

    def test_create_table_default(self):
        @named_tuple_table_class
        class A(NamedTuple):
            a: int
            b: int = 0
            c: str = ''
            d: bool = True
            e: bool = False
            f: str = None

        with Connection(debug=True) as conn:
            sqlp.create_table(A)
            self.assert_sql_state_equal("""
            CREATE TABLE A (
                [a] INTEGER NOT NULL ,
                [b] INTEGER NOT NULL DEFAULT 0 ,
                [c] TEXT NOT NULL DEFAULT '' ,
                [d] BOOLEAN NOT NULL DEFAULT True ,
                [e] BOOLEAN NOT NULL DEFAULT False ,
                [f] TEXT DEFAULT NULL
            )
            """, conn.table_schema(A))
            sqlp.insert_into(A).submit([A(1)])

            result = sqlp.select_from(A).fetchone()
            self.assertEqual(result, A(1, 0, '', True, False, None))

    def test_table_field_bool_casting(self):
        @named_tuple_table_class
        class A(NamedTuple):
            a: int
            b: bool

        with Connection(debug=True):
            sqlp.create_table(A)
            sqlp.insert_into(A).submit([A(0, False), A(1, True)])
            self.assertEqual(A(0, False), sqlp.select_from(A).where(A.a == 0).fetchone())
            self.assertEqual(A(1, True), sqlp.select_from(A).where(A.a == 1).fetchone())
            self.assertEqual((False,), sqlp.select_from(A.b).where(A.a == 0).fetchone())
            self.assertEqual((True,), sqlp.select_from(A.b).where(A.a == 1).fetchone())

    def test_table_field_path_casting(self):
        @named_tuple_table_class
        class A(NamedTuple):
            a: Annotated[Path, PRIMARY]

        with Connection(debug=True) as conn:
            sqlp.create_table(A)
            self.assert_sql_state_equal("""
            CREATE TABLE A (
                [a] TEXT NOT NULL PRIMARY KEY
            )
            """, conn.table_schema(A))

            sqlp.insert_into(A).submit([A(Path('test.txt'))])

            result = sqlp.select_from(A).fetchone()
            self.assertEqual(result, A(Path('test.txt')))

    def test_table_property(self):
        @named_tuple_table_class
        class A(NamedTuple):
            a: int
            b: int

            @property
            def c(self) -> int:
                return self.a + self.b

        with Connection(debug=True) as conn:
            sqlp.create_table(A)

            self.assertEqual(6, A(2, 4).c)

            sqlp.insert_into(A).submit([
                A(0, 1),
                A(1, 2),
                A(2, 3),
            ])

            results = sqlp.select_from(A.a, A.b, A.c).fetchall()
            self.assertListEqual([
                (0, 1, 1), (1, 2, 3), (2, 3, 5)
            ], results)

    def test_table_property_sqlp_call(self):
        @named_tuple_table_class
        class A(NamedTuple):
            a: str
            b: str

            @property
            def c(self) -> str:
                return sqlp.concat(self.a, self.b)

        with Connection(debug=True) as conn:
            sqlp.create_table(A)

            value = A('2', '4').c
            self.assertIsInstance(value, str)
            self.assertEqual('24', value)

            sqlp.insert_into(A).submit([
                A('0', '1'),
                A('1', '2'),
                A('2', '3'),
            ])

            results = sqlp.select_from(A.a, A.b, A.c).fetchall()
            self.assertListEqual([
                ('0', '1', '01'), ('1', '2', '12'), ('2', '3', '23')
            ], results)



if __name__ == '__main__':
    unittest.main()
