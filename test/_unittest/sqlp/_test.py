import unittest
from pathlib import Path
from typing import Optional, TypeVar

from neuralib.sqlp.connection import Connection
from neuralib.sqlp.stat import SqlStat

__all__ = ['SqlTestCase']

T = TypeVar('T')


class SqlTestCase(unittest.TestCase):
    source_database: Optional[Path] = Path(__file__).with_name('chinook.db')
    connection: Connection

    @classmethod
    def setUpClass(cls):
        if cls.source_database is not None:
            if not cls.source_database.exists():
                raise RuntimeError('wget https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip; unzip chinook.zip')
            cls.connection = Connection(cls.source_database)
        else:
            cls.connection = Connection()

        cls.connection.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.connection.__exit__(None, None, None)

    def assertSqlExeEqual(self, raw_sql: str, stat: SqlStat[T], parameter=()) -> list[T]:
        connection = self.connection
        r1 = connection.execute(raw_sql, parameter).fetchall()
        r2 = connection.execute(stat, parameter).fetchall()
        self.assertListEqual(r1, r2)
        return r2
