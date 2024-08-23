import shutil
import unittest
from pathlib import Path
from typing import Annotated, NamedTuple, Optional, Union, TypeVar

from _unittest.sqlp._test import SqlTestCase
from _unittest.sqlp._tracks import *
from neuralib import sqlp
from neuralib.sqlp import Connection
from neuralib.sqlp.stat import SqlStat, Cursor

T = TypeVar('T')


class ChangeTest(SqlTestCase):
    ref_database: Optional[Path]
    tmp_database: Optional[Path]

    @classmethod
    def setUpClass(cls):
        cls.ref_database = Path('/tmp/chinook1.db')
        cls.tmp_database = Path('/tmp/chinook2.db')

    @classmethod
    def tearDownClass(cls):
        if cls.ref_database is not None:
            cls.ref_database.unlink(missing_ok=True)
        if cls.tmp_database is not None:
            cls.tmp_database.unlink(missing_ok=True)

    def setUp(self):
        if self.ref_database is not None:
            shutil.copyfile(self.source_database, self.ref_database)
            self.connection_ref = Connection(self.ref_database)
        else:
            self.connection_ref = Connection()

        if self.tmp_database is not None:
            shutil.copyfile(self.source_database, self.tmp_database)
            self.connection = Connection(self.tmp_database, debug=True)
        else:
            self.connection = Connection(debug=True)

        self.connection.__enter__()

    def tearDown(self):
        self.connection.__exit__(None, None, None)
        if self.ref_database is not None:
            self.ref_database.unlink(missing_ok=True)
        if self.tmp_database is not None:
            self.tmp_database.unlink(missing_ok=True)

    def execute_both(self, raw_sql: str, parameter=()):
        self.connection_ref.execute(raw_sql, parameter)
        self.connection.execute(raw_sql, parameter)

    def assertSqlExeEqual(self, raw_sql: str, stat: Union[SqlStat[T], Cursor[T]], parameter=()) -> list[T]:
        r1 = self.connection_ref.execute(raw_sql, parameter).fetchall()

        if isinstance(stat, Cursor):
            r2 = stat.fetchall()
        else:
            r2 = self.connection.execute(stat, parameter).fetchall()

        self.assertListEqual(r1, r2)
        return r2

    @classmethod
    def is_compiled_with(cls, opt: str) -> bool:
        return Connection(cls.source_database).sqlite_compileoption_used(opt)

    @classmethod
    def skip_without_compile_with(cls, opt: str):
        return unittest.skipIf(not cls.is_compiled_with(opt), f'!{opt}')


class InsertTest(ChangeTest):
    """https://www.sqlitetutorial.net/sqlite-insert/"""

    def assert_latest_artists(self, n: int = 1) -> list[Artists]:
        return self.assertSqlExeEqual("""\
        SELECT
            ArtistId,
            Name
        FROM
            Artists
        ORDER BY
            ArtistId DESC
        LIMIT 1;
        """, sqlp.select_from(Artists.ArtistId, Artists.Name).order_by(sqlp.desc(Artists.ArtistId)).limit(n))

    def test_insert_one(self):
        self.assertSqlExeEqual("""\
        INSERT INTO artists (name)
        VALUES('Bud Powell');
        """, sqlp.insert_into(Artists.Name).submit([('Bud Powell',)]))

        self.assert_latest_artists(1)

    def test_insert_default(self):
        self.assertSqlExeEqual("""\
        INSERT INTO artists DEFAULT VALUES;
        """, sqlp.insert_into(Artists).defaults())

        self.assert_latest_artists(1)

    def test_insert_from_select(self):
        @sqlp.named_tuple_table_class
        class ArtistsBackup(NamedTuple):
            ArtistsId: Annotated[int, sqlp.PRIMARY(auto_increment=True)]
            Name: str

        self.assertSqlExeEqual("""\
        CREATE TABLE ArtistsBackup(
           ArtistId INTEGER PRIMARY KEY AUTOINCREMENT,
           Name NVARCHAR
        );
        """, sqlp.create_table(ArtistsBackup))
        self.assertSqlExeEqual("""\
        INSERT INTO ArtistsBackup 
        SELECT ArtistId, Name
        FROM artists;
        """, sqlp.insert_into(ArtistsBackup).select_from(Artists.ArtistId, Artists.Name))
        self.assertSqlExeEqual("""SELECT * FROM ArtistsBackup;""",
                               sqlp.select_from(ArtistsBackup))


class UpdateTest(ChangeTest):
    """ https://www.sqlitetutorial.net/sqlite-update/ """

    def test_update_one(self):
        self.assertSqlExeEqual("""\
        UPDATE employees
        SET lastname = 'Smith'
        WHERE employeeid = 3;
        """, sqlp.update(Employees, Employees.LastName == 'Smith').where(Employees.EmployeeId == 3))

        self.assertSqlExeEqual("""\
        SELECT
            employeeid,
            firstname,
            lastname,
            title,
            email
        FROM
            employees
        WHERE
            employeeid = 3;
        """, sqlp.select_from(Employees.EmployeeId, Employees.FirstName, Employees.LastName, Employees.Title, Employees.Email).where(
            Employees.EmployeeId == 3
        ))

    def test_update_multiple(self):
        self.assertSqlExeEqual("""\
        UPDATE employees
        SET city = 'Toronto',
            state = 'ON',
            postalcode = 'M5P 2N7'
        WHERE
            employeeid = 4;
        """, sqlp.update(
            Employees,
            Employees.City == 'Toronto',
            Employees.State == 'ON',
            Employees.PostalCode == 'M5P 2N7'
        ).where(Employees.EmployeeId == 4))

        self.assertSqlExeEqual("""\
        SELECT
            employeeid,
            firstname,
            lastname,
            title,
            email
        FROM
            employees
        WHERE
            employeeid = 4;
        """, sqlp.select_from(Employees.EmployeeId, Employees.FirstName, Employees.LastName, Employees.Title, Employees.Email).where(
            Employees.EmployeeId == 4
        ))

    @ChangeTest.skip_without_compile_with('SQLITE_ENABLE_UPDATE_DELETE_LIMIT')
    def test_update_with_expr(self):
        self.assertSqlExeEqual("""\
        UPDATE employees
        SET email = LOWER( firstname || "." || lastname || "@chinookcorp.com" )
        ORDER BY firstname
        LIMIT 1;
        """, sqlp.update(
            Employees,
            Employees.Email == sqlp.lower(sqlp.concat(Employees.FirstName, '.', Employees.LastName, '@chinookcorp.com'))
        ).order_by(Employees.FirstName).limit(1))

        self.assertSqlExeEqual("""
        SELECT 
            employeeid,
            firstname,
            lastname,
            email
        FROM
            employees
        ORDER BY
            firstname
        LIMIT 5;
        """, sqlp.select_from(Employees.EmployeeId, Employees.FirstName, Employees.LastName, Employees.Email).order_by(Employees.FirstName).limit(5))

    def test_update_all(self):
        self.assertSqlExeEqual("""\
        UPDATE employees
        SET email = LOWER(
            firstname || "." || lastname || "@chinookcorp.com"
        );
        """, sqlp.update(
            Employees,
            Employees.Email == sqlp.lower(sqlp.concat(Employees.FirstName, '.', Employees.LastName, '@chinookcorp.com'))
        ))

        self.assertSqlExeEqual("""
        SELECT 
            employeeid,
            firstname,
            lastname,
            email
        FROM
            employees
        ORDER BY
            firstname
        """, sqlp.select_from(Employees.EmployeeId, Employees.FirstName, Employees.LastName, Employees.Email).order_by(Employees.FirstName))


class UpdateFromTest(ChangeTest):
    """
    https://www.sqlitetutorial.net/sqlite-update-from/
    """

    @classmethod
    def setUpClass(cls):
        cls.ref_database = None
        cls.tmp_database = None

    def test_update_from(self):
        @sqlp.named_tuple_table_class
        class Inventory(NamedTuple):
            item_id: Annotated[int, sqlp.PRIMARY]
            item_name: str
            quantity: int

        @sqlp.named_tuple_table_class
        class Sales(NamedTuple):
            sales_id: Annotated[Optional[int], sqlp.PRIMARY]
            item_id: Optional[int]
            quantity_sold: Optional[int]

            # sales_at: datetime.datetime

            @sqlp.foreign(Inventory)
            def _inventory(self):
                return self.item_id

        self.assertSqlExeEqual("""\
        CREATE TABLE inventory (
            item_id INTEGER PRIMARY KEY,
            item_name TEXT NOT NULL,
            quantity INTEGER NOT NULL
        );
        """, sqlp.create_table(Inventory))

        self.assertSqlExeEqual("""\
        CREATE TABLE sales (
            sales_id INTEGER PRIMARY KEY,
            item_id INTEGER,
            quantity_sold INTEGER,
            -- sales_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (item_id) REFERENCES inventory (item_id)
        );
        """, sqlp.create_table(Sales))

        self.execute_both("""\
        INSERT INTO
          inventory (item_id, item_name, quantity)
        VALUES
          (1, 'Item A', 100),
          (2, 'Item B', 150),
          (3, 'Item C', 200);
        """)
        self.execute_both("""\
        INSERT INTO
          sales (item_id, quantity_sold)
        VALUES
          (1, 20),
          (1, 30),
          (2, 25),
          (3, 50);
        """)

        daily = sqlp.select_from(
            sqlp.sum(Sales.quantity_sold) @ 'qty',
            Sales.item_id,
            from_table=Sales
        ).group_by(Sales.item_id) @ 'daily'

        self.assertSqlExeEqual("""\
        UPDATE inventory
        SET
          quantity = quantity - daily.qty
        FROM
          (
            SELECT
              SUM(quantity_sold) AS qty,
              item_id
            FROM
              sales
            GROUP BY
              item_id
          ) AS daily
        WHERE
          inventory.item_id = daily.item_id;
        """, sqlp.update(
            Inventory,
            Inventory.quantity == Inventory.quantity - daily.qty
        ).from_(daily).where(Inventory.item_id == daily.item_id))


class DeleteTest(ChangeTest):
    """
    https://www.sqlitetutorial.net/sqlite-delete/
    """

    def test_delete_by_id(self):
        self.assertSqlExeEqual("""\
        DELETE FROM artists
        WHERE artistid = 1;
        """, sqlp.delete_from(Artists).where(Artists.ArtistId == 1))
        self.assertSqlExeEqual("""\
        SELECT artistid, name FROM artists WHERE artistid < 10;
        """, sqlp.select_from(Artists.ArtistId, Artists.Name).where(Artists.ArtistId < 10))

    def test_delete_by_pattern(self):
        self.assertSqlExeEqual("""\
        DELETE FROM artists
        WHERE name LIKE '%Santana%';
        """, sqlp.delete_from(Artists).where(sqlp.like(Artists.Name, '%Santana%')))
        self.assertSqlExeEqual("""\
        SELECT * FROM artists;
        """, sqlp.select_from(Artists))


@sqlp.named_tuple_table_class
class Positions(NamedTuple):
    id: Annotated[int, sqlp.PRIMARY]
    title: Annotated[str, sqlp.UNIQUE]
    min_salary: Optional[float]


class ReplaceTest(ChangeTest):
    """https://www.sqlitetutorial.net/sqlite-replace-statement/"""

    @classmethod
    def setUpClass(cls):
        cls.ref_database = None
        cls.tmp_database = None

    def setUp(self):
        super().setUp()
        self.execute_both("""\
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY,
            title TEXT UNIQUE NOT NULL,
            min_salary NUMERIC
        );
        """)
        self.execute_both("""\
        INSERT INTO positions (title, min_salary)
        VALUES ('DBA', 120000),
               ('Developer', 100000),
               ('Architect', 150000);
        """)
        # self.execute_both("""\
        # CREATE UNIQUE INDEX idx_positions_title
        # ON positions (title);
        # """)

    def test_setup(self):
        self.assertSqlExeEqual("""SELECT * FROM positions;""",
                               sqlp.select_from(Positions))

    def test_insert(self):
        self.assertSqlExeEqual("""\
        REPLACE INTO positions (title, min_salary)
        VALUES('Full Stack Developer', 140000);
        """, sqlp.replace_into(Positions.title, Positions.min_salary).submit(
            [('Full Stack Developer', 140000)]
        ))
        self.assertSqlExeEqual("""\
        SELECT
            *
        FROM
            positions;
        """, sqlp.select_from(Positions))

    def test_replace(self):
        self.assertSqlExeEqual("""\
        REPLACE INTO positions (title, min_salary)
        VALUES('DBA', 170000);
        """, sqlp.replace_into(Positions.title, Positions.min_salary).submit(
            [('DBA', 170000)]
        ))
        self.assertSqlExeEqual("""\
        SELECT
            *
        FROM
            positions;
        """, sqlp.select_from(Positions))


@sqlp.named_tuple_table_class
class SearchStats(NamedTuple):
    id: Annotated[Optional[int], sqlp.PRIMARY]
    keyword: Annotated[str, sqlp.UNIQUE]
    search_count: int = 1


class UpsertTest(ChangeTest):
    """https://www.sqlitetutorial.net/sqlite-upsert/"""

    @classmethod
    def setUpClass(cls):
        cls.ref_database = None
        cls.tmp_database = None

    def setUp(self):
        super().setUp()
        self.execute_both("""\
        CREATE TABLE SearchStats(
           id INTEGER PRIMARY KEY,
           keyword TEXT UNIQUE NOT NULL,
           search_count INT NOT NULL DEFAULT 1   
        );
        """)
        self.execute_both("""\
        INSERT INTO SearchStats(keyword)
        VALUES('SQLite');
        """)

    def test_no_nothing(self):
        self.assertSqlExeEqual("""\
        INSERT INTO SearchStats(keyword)
        VALUES ('SQLite')
        ON CONFLICT (keyword)
        DO NOTHING;
        """, sqlp.insert_into(SearchStats.keyword).on_conflict(
            SearchStats.keyword
        ).do_nothing(
        ).submit([('SQLite',)]))
        ret = self.assertSqlExeEqual("""SELECT * FROM SearchStats;""",
                                     sqlp.select_from(SearchStats))
        print(ret)

    def test_update(self):
        self.assertSqlExeEqual("""\
        INSERT INTO SearchStats(keyword)
        VALUES ('SQLite')
        ON CONFLICT (keyword)
        DO 
           UPDATE 
           SET search_count = search_count + 1;
        """, sqlp.insert_into(SearchStats.keyword).on_conflict(
            SearchStats.keyword
        ).do_update(
            SearchStats.search_count == SearchStats.search_count + 1
        ).submit([('SQLite',)]))
        ret = self.assertSqlExeEqual("""SELECT * FROM SearchStats;""",
                                     sqlp.select_from(SearchStats))
        print(ret)

    def test_update_where(self):
        excluded = sqlp.excluded(SearchStats)

        self.assertSqlExeEqual("""\
        INSERT INTO SearchStats(keyword, search_count)
        VALUES ('SQLite', -1)
        ON CONFLICT (keyword)
        DO 
           UPDATE 
           SET search_count = excluded.search_count
        WHERE
            excluded.search_count > 0;
        """, sqlp.insert_into(SearchStats.keyword, SearchStats.search_count).on_conflict(
            SearchStats.keyword
        ).do_update(
            SearchStats.search_count == excluded.search_count,
            where=excluded.search_count > 0
        ).submit([('SQLite', -1)]))

        ret = self.assertSqlExeEqual("""SELECT * FROM SearchStats;""",
                                     sqlp.select_from(SearchStats))
        print(ret)


@sqlp.named_tuple_table_class
class Books(NamedTuple):
    id: Annotated[Optional[int], sqlp.PRIMARY]
    title: str
    isbn: str
    # use str for testing
    release_date: str  # datetime.date


class ReturningTest(ChangeTest):
    """https://www.sqlitetutorial.net/sqlite-returning/"""

    @classmethod
    def setUpClass(cls):
        cls.ref_database = None
        cls.tmp_database = None

    def setUp(self):
        super().setUp()

        self.execute_both("""\
        CREATE TABLE books(
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            isbn TEXT NOT NULL,
            release_date DATE
        );
        """)

        self.execute_both("""\
        INSERT INTO books(title, isbn, release_date)
        VALUES('The Catcher in the Rye', '9780316769488', '1951-07-16');
        """)

    def test_insert_return(self):
        self.assertSqlExeEqual("""\
        INSERT INTO books(title, isbn, release_date)
        VALUES ('The Great Gatsby', '9780743273565', '1925-04-10')
        RETURNING id;
        """, sqlp.insert_into(
            Books.title, Books.isbn, Books.release_date
        ).returning(Books.id).submit(
            [('The Great Gatsby', '9780743273565', '1925-04-10')]
        ))

    def test_insert_return_alias(self):
        ret = self.assertSqlExeEqual("""\
        INSERT INTO books(title, isbn, release_date)
        VALUES ('The Great Gatsby', '9780743273565', '1925-04-10')
        RETURNING 
           id AS book_id, 
           strftime('%Y', release_date) AS year;
        """, sqlp.insert_into(
            Books.title, Books.isbn, Books.release_date
        ).returning(
            Books.id @ 'book_id',
            sqlp.strftime('%Y', Books.release_date) @ 'year'
        ).submit(
            [('The Great Gatsby', '9780743273565', '1925-04-10')]
        ))
        print(ret)

    def test_insert_return_all(self):
        # TODO only support return one data
        self.assertSqlExeEqual("""\
        INSERT INTO books (title, isbn, release_date) 
        VALUES
            --('Pride and Prejudice', '9780141439518', '1813-01-28'),
            ('The Lord of the Rings', '9780618640157', '1954-07-29')
        RETURNING *;
        """, sqlp.insert_into(
            Books.title, Books.isbn, Books.release_date
        ).returning().submit([
            # ('Pride and Prejudice', '9780141439518', '1813-01-28'),
            ('The Lord of the Rings', '9780618640157', '1954-07-29')
        ]))

    def test_update_return(self):
        self.assertSqlExeEqual("""\
        UPDATE books
        SET isbn = '0141439512'
        WHERE id = 1
        RETURNING *;
        """, sqlp.update(Books, Books.isbn == '0141439512').where(Books.id == 1).returning())

    def test_update_all_return(self):
        self.assertSqlExeEqual("""\
        UPDATE books
        SET title = UPPER(title)
        RETURNING *;
        """, sqlp.update(Books, Books.title == sqlp.upper(Books.title)).returning())

    def test_delete_return(self):
        self.assertSqlExeEqual("""\
        DELETE FROM books
        WHERE id = 1
        RETURNING *;
        """, sqlp.delete_from(Books).where(Books.id == 1).returning())

    def test_delete_all_return(self):
        self.assertSqlExeEqual("""\
        DELETE FROM books
        RETURNING *;
        """, sqlp.delete_from(Books).returning())


if __name__ == '__main__':
    unittest.main()
