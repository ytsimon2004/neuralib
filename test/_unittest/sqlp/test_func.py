import unittest
from typing import NamedTuple, Any, Optional

from _test import SqlTestCase
from _tracks import *
from neuralib import sqlp


@sqlp.named_tuple_table_class
class AvgTest(NamedTuple):
    val: Optional[Any]


class FuncAvgTest(SqlTestCase):
    """https://www.sqlitetutorial.net/sqlite-avg/"""

    @classmethod
    def setUpClass(cls):
        cls.source_database = None
        super().setUpClass()

    def setUp(self):
        sqlp.create_table(AvgTest).submit()

        self.connection.execute("""\
        INSERT INTO AvgTest (val)
        VALUES
         (1),
         (2),
         (10.1),
         (20.5),
         ('8'),
         ('B'),
         (NULL),
         (x'0010'),
         (x'0011');
        """)

    def test_setup(self):
        ret = self.assertSqlExeEqual("""SELECT rowid, val FROM AvgTest;""",
                                     sqlp.select_from(sqlp.ROWID, AvgTest.val))
        print(ret)

    def test_avg(self):
        self.assertSqlExeEqual("""\
        SELECT
            avg(val)
        FROM
            AvgTest
        WHERE
            rowid < 5;
        """, sqlp.select_from(sqlp.avg(AvgTest.val)).where(sqlp.ROWID < 5))

    def test_avg_on_null(self):
        self.assertSqlExeEqual("""\
        SELECT
            avg(val)
        FROM
            AvgTest;
        """, sqlp.select_from(sqlp.avg(AvgTest.val)))

    def test_avg_distinct(self):
        self.assertSqlExeEqual("""\
        SELECT
            avg(DISTINCT val)
        FROM
            AvgTest;
        """, sqlp.select_from(sqlp.avg(AvgTest.val).distinct()))


class FuncAvgExampleTest(SqlTestCase):
    """https://www.sqlitetutorial.net/sqlite-avg/"""

    def test_avg_all(self):
        self.assertSqlExeEqual("""\
        SELECT
            avg(milliseconds)
        FROM
            tracks;
        """, sqlp.select_from(sqlp.avg(Tracks.Milliseconds)))

    def test_avg_group_by(self):
        self.assertSqlExeEqual("""\
        SELECT
            albumid,
            avg(milliseconds)
        FROM
            tracks
        GROUP BY
            albumid;
        """, sqlp.select_from(Tracks.AlbumId, sqlp.avg(Tracks.Milliseconds)).group_by(Tracks.AlbumId))

    def test_avg_join(self):
        self.assertSqlExeEqual("""\
        SELECT
            tracks.AlbumId,
            Title,
            round(avg(Milliseconds), 2) avg_length
        FROM
            tracks
        INNER JOIN albums ON albums.AlbumId = tracks.albumid
        GROUP BY
            tracks.albumid;
        """, sqlp.select_from(
            Tracks.AlbumId,
            Albums.Title,
            sqlp.round(sqlp.avg(Tracks.Milliseconds), 2) @ 'avg_length',
            from_table=Tracks
        ).join(Albums, by='inner').on(
            Albums.AlbumId == Tracks.AlbumId
        ).group_by(Tracks.AlbumId))

    def test_avg_join_having(self):
        self.assertSqlExeEqual("""\
        SELECT
            tracks.albumid,
            title,
            round(avg(milliseconds),2)  avg_length
        FROM
            tracks
        INNER JOIN albums ON albums.AlbumId = tracks.albumid
        GROUP BY
            tracks.albumid
        HAVING
            avg_length BETWEEN 100000 AND 200000;
        """, sqlp.select_from(
            Tracks.AlbumId,
            Albums.Title,
            (avg_length := sqlp.round(sqlp.avg(Tracks.Milliseconds), 2) @ 'avg_length'),
            from_table=Tracks
        ).join(Albums, by='inner').on(
            Albums.AlbumId == Tracks.AlbumId
        ).group_by(Tracks.AlbumId).having(
            sqlp.between(avg_length, 100000, 200000)
        ))


@sqlp.named_tuple_table_class
class CountTest(NamedTuple):
    c: Optional[int]


class FuncCountTest(SqlTestCase):
    """https://www.sqlitetutorial.net/sqlite-count-function/"""

    @classmethod
    def setUpClass(cls):
        cls.source_database = None
        super().setUpClass()

    def setUp(self):
        sqlp.create_table(CountTest).submit()

        self.connection.execute("""\
           INSERT INTO CountTest (c)
           VALUES(1),(2),(3),(null),(3);
           """)

    def test_setup(self):
        self.assertSqlExeEqual("""SELECT * FROM CountTest;""",
                               sqlp.select_from(CountTest))

    def test_count_all(self):
        self.assertSqlExeEqual("""SELECT COUNT(*) FROM CountTest;""",
                               sqlp.select_from(sqlp.count(), from_table=CountTest))

    def test_count_non_null(self):
        self.assertSqlExeEqual("""SELECT COUNT(c) FROM CountTest;""",
                               sqlp.select_from(sqlp.count(CountTest.c)))

    def test_count_distinct(self):
        self.assertSqlExeEqual("""SELECT COUNT(DISTINCT c) FROM CountTest;""",
                               sqlp.select_from(sqlp.count(CountTest.c).distinct()))


class FuncCountExampleTest(SqlTestCase):
    """https://www.sqlitetutorial.net/sqlite-count-function/"""

    def test_count_all(self):
        self.assertSqlExeEqual("""\
        SELECT count(*)
        FROM tracks;
        """, sqlp.select_from(sqlp.count(), from_table=Tracks))

    def test_count_where(self):
        self.assertSqlExeEqual("""\
        SELECT count(*)
        FROM tracks
        WHERE albumid = 10;
        """, sqlp.select_from(sqlp.count(), from_table=Tracks).where(Tracks.AlbumId == 10))

    def test_count_group_by(self):
        self.assertSqlExeEqual("""\
        SELECT count(*)
        FROM tracks
        GROUP BY
            albumid;
        """, sqlp.select_from(sqlp.count(), from_table=Tracks).group_by(Tracks.AlbumId))

    def test_count_having(self):
        self.assertSqlExeEqual("""\
        SELECT 
            albumid,
            count(*)
        FROM tracks
        GROUP BY
            albumid
        HAVING COUNT(*) > 25
        """, sqlp.select_from(
            Tracks.AlbumId, sqlp.count()
        ).group_by(Tracks.AlbumId).having(
            sqlp.count() > 25
        ))

    def test_count_join(self):
        self.assertSqlExeEqual("""\
        SELECT
            tracks.albumid, 
            title, 
            COUNT(*)
        FROM
            tracks
        INNER JOIN albums ON
            albums.albumid = tracks.albumid
        GROUP BY
            tracks.albumid
        HAVING
            COUNT(*) > 25
        ORDER BY
            COUNT(*) DESC;
        """, sqlp.select_from(
            Tracks.AlbumId,
            Albums.Title,
            sqlp.count()
        ).join(Albums, by='inner').on(
            Tracks.AlbumId == Albums.AlbumId
        ).group_by(
            Tracks.AlbumId
        ).having(
            sqlp.count() > 25
        ).order_by(
            sqlp.desc(sqlp.count())
        ))


    def test_count_distinct(self):
        self.assertSqlExeEqual("""\
        SELECT COUNT(title)
        FROM employees;
        """, sqlp.select_from(sqlp.count(Employees.Title)))

        self.assertSqlExeEqual("""\
        SELECT COUNT(DISTINCT title)
        FROM employees;
        """, sqlp.select_from(sqlp.count(Employees.Title).distinct()))


class FuncMinMaxTest(SqlTestCase):
    """
    https://www.sqlitetutorial.net/sqlite-max/
    https://www.sqlitetutorial.net/sqlite-min/
    """

    def test_max(self):
        self.assertSqlExeEqual("""\
        SELECT MAX(bytes) FROM tracks;
        """, sqlp.select_from(sqlp.max(Tracks.Bytes)))

    def test_max_subquery(self):
        self.assertSqlExeEqual("""\
        SELECT
            TrackId,
            Name,
            Bytes
        FROM
            tracks
        WHERE
            Bytes = (SELECT MAX(Bytes) FROM tracks);
        """, sqlp.select_from(
            Tracks.TrackId,
            Tracks.Name,
            Tracks.Bytes
        ).where(
            Tracks.Bytes == sqlp.select_from(sqlp.max(Tracks.Bytes))
        ))

    def test_max_group_by(self):
        self.assertSqlExeEqual("""\
        SELECT
            AlbumId,
            MAX(bytes)
        FROM
            tracks
        GROUP BY
            AlbumId;
        """, sqlp.select_from(Tracks.AlbumId, sqlp.max(Tracks.Bytes)).group_by(Tracks.AlbumId))

    def test_max_having(self):
        self.assertSqlExeEqual("""\
        SELECT
            AlbumId,
            MAX(bytes)
        FROM
            tracks
        GROUP BY
            AlbumId
        HAVING MAX(bytes) > 6000000;
        """, sqlp.select_from(
            Tracks.AlbumId, sqlp.max(Tracks.Bytes)
        ).group_by(Tracks.AlbumId).having(
            sqlp.max(Tracks.Bytes) > 6000000
        ))


class FuncSumTest(SqlTestCase):
    """
    https://www.sqlitetutorial.net/sqlite-sum/
    """

    def test_sum(self):
        self.assertSqlExeEqual("""\
        SELECT
           SUM(milliseconds)
        FROM
           tracks;
        """, sqlp.select_from(sqlp.sum(Tracks.Milliseconds)))

    def test_sum_group_by(self):
        self.assertSqlExeEqual("""\
        SELECT
            AlbumId,
            SUM(milliseconds)
        FROM
            tracks
        GROUP BY
            AlbumId;
        """, sqlp.select_from(Tracks.AlbumId, sqlp.sum(Tracks.Milliseconds)).group_by(Tracks.AlbumId))

    def test_sum_join(self):
        self.assertSqlExeEqual("""\
        SELECT
           tracks.albumid,
           title, 
           SUM(milliseconds)
        FROM
           tracks
        INNER JOIN albums ON albums.albumid = tracks.albumid
        GROUP BY
           tracks.albumid, 
           title;
        """, sqlp.select_from(
            Tracks.AlbumId, Albums.Title, sqlp.sum(Tracks.Milliseconds)
        ).join(Albums, by='inner').by(
            Tracks._albums
        ).group_by(
            Tracks.AlbumId, Albums.Title
        ))

    def test_sum_join_having(self):
        self.assertSqlExeEqual("""\
        SELECT
           tracks.albumid,
           title, 
           SUM(milliseconds)
        FROM
           tracks
        INNER JOIN albums ON albums.albumid = tracks.albumid
        GROUP BY
           tracks.albumid, 
           title
        HAVING
           SUM(milliseconds) > 1000000;
        """, sqlp.select_from(
            Tracks.AlbumId, Albums.Title, sqlp.sum(Tracks.Milliseconds)
        ).join(Albums, by='inner').by(
            Tracks._albums
        ).group_by(
            Tracks.AlbumId, Albums.Title
        ).having(
            sqlp.sum(Tracks.Milliseconds) > 1000000
        ))


# https://www.sqlitetutorial.net/sqlite-group_concat/

if __name__ == '__main__':
    unittest.main()
