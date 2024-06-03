import re
import sqlite3
import unittest
from pathlib import Path
from typing import NamedTuple, Annotated, Optional

from neuralib.sqlp import *
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

    @foreign(Person.name)
    def _person(self):
        return self.person

    @foreign(Bank.name)
    def _bank(self):
        return self.bank


class SqlpTableTest(unittest.TestCase):
    conn: Connection

    @classmethod
    def setUpClass(cls):
        cls.conn = Connection(debug=True)
        with cls.conn:
            create_table(Person)
            create_table(Bank)
            create_table(Account)

            insert_into(Person).submit([
                Person('Alice', 18),
                Person('Bob', 20),
            ])

            insert_into(Bank).submit([
                Bank('V'),
                Bank('M'),
            ])

            insert_into(Account).submit([
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
            insert_into(Person).submit([Person('baby', 1)])

    def test_select_person(self):
        results = select_from(Person).order_by(Person.name).fetchall()

        self.assertListEqual([
            Person('Alice', 18),
            Person('Bob', 20),
        ], results)

    def test_select_with_literal(self):
        results = select_from(Person.name, Person.age, 1).order_by(Person.name).fetchall()

        self.assertListEqual([
            ('Alice', 18, 1),
            ('Bob', 20, 1),
        ], results)

    def test_where(self):
        results = select_from(Account).where(
            Account.bank == 'V'
        ).fetchall()

        self.assertListEqual([
            Account('V', 'Alice', 1000),
            Account('V', 'Bob', 2000),
        ], results)

    def test_update(self):
        insert_into(Account, policy='REPLACE').submit([Account('K', 'Eve', 0)])
        self.assertEqual(Account('K', 'Eve', 0), select_from(Account).where(Account.bank == 'K').fetchone())
        update(Account, Account.money == 100).where(
            Account.bank == 'K'
        ).submit(commit=True)
        self.assertEqual(Account('K', 'Eve', 100), select_from(Account).where(Account.bank == 'K').fetchone())

    def test_delete(self):
        insert_into(Account, policy='REPLACE').submit([Account('K', 'Eve', 0)])
        self.assertEqual(Account('K', 'Eve', 0), select_from(Account).where(Account.bank == 'K').fetchone())
        delete_from(Account).where(Account.bank == 'K').submit(commit=True)
        self.assertIsNone(select_from(Account).where(Account.bank == 'K').fetchone())

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
        person = table_foreign_field(Account, Account._person)
        self.assertIsInstance(person, ForeignConstraint)
        self.assertEqual(person.name, '_person')
        self.assertEqual(person.foreign_table, Person)
        self.assertListEqual(person.foreign_fields, ['name'])
        self.assertEqual(person.table, Account)
        self.assertListEqual(person.fields, ['person'])

    def test_map_foreign(self):
        from neuralib.sqlp.util import map_foreign
        results = map_foreign(Account('V', 'Alice', 1000), Account._person).fetchall()
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
            create_table(A)
            self.assert_sql_state_equal("""
        CREATE TABLE A (
            a INT NOT NULL ,
            b INT 
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
            create_table(A)
            self.assert_sql_state_equal("""
            CREATE TABLE A (
                a INT NOT NULL ,
                b INT NOT NULL DEFAULT 0 ,
                c TEXT NOT NULL DEFAULT '' ,
                d BOOLEAN NOT NULL DEFAULT True ,
                e BOOLEAN NOT NULL DEFAULT False ,
                f TEXT DEFAULT NULL 
            )
            """, conn.table_schema(A))
            insert_into(A).submit([A(1)])

            result = select_from(A).fetchone()
            self.assertEqual(result, A(1, 0, '', True, False, None))

    def test_table_field_bool_casting(self):
        @named_tuple_table_class
        class A(NamedTuple):
            a: int
            b: bool

        with Connection(debug=True):
            create_table(A)
            insert_into(A).submit([A(0, False), A(1, True)])
            self.assertEqual(A(0, False), select_from(A).where(A.a == 0).fetchone())
            self.assertEqual(A(1, True), select_from(A).where(A.a == 1).fetchone())
            self.assertEqual((False,), select_from(A.b).where(A.a == 0).fetchone())
            self.assertEqual((True,), select_from(A.b).where(A.a == 1).fetchone())

    def test_table_field_path_casting(self):
        @named_tuple_table_class
        class A(NamedTuple):
            a: Annotated[Path, PRIMARY]

        with Connection(debug=True) as conn:
            create_table(A)
            self.assert_sql_state_equal("""
                   CREATE TABLE A (
                       a TEXT NOT NULL ,
                       PRIMARY KEY ( a ) 
                   )
                   """, conn.table_schema(A))

            insert_into(A).submit([A(Path('test.txt'))])

            result = select_from(A).fetchone()
            self.assertEqual(result, A(Path('test.txt')))


if __name__ == '__main__':
    unittest.main()
