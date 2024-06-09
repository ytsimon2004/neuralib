import datetime
import unittest

from _test import SqlTestCase
from _tracks import *
from neuralib import sqlp


class SelectTest(SqlTestCase):
    """
    https://www.sqlitetutorial.net/sqlite-select/
    https://www.sqlitetutorial.net/sqlite-distinct/

    """

    def test_select(self):
        self.assertSqlExeEqual("SELECT trackid, name, composer, unitprice  FROM tracks;",
                               sqlp.select_from(Tracks.TrackId, Tracks.Name, Tracks.Composer, Tracks.UnitPrice))

    def test_select_all(self):
        self.assertSqlExeEqual("SELECT * FROM tracks;", sqlp.select_from(Tracks))

    def test_select_distinct(self):
        self.assertSqlExeEqual("SELECT DISTINCT city FROM customers ORDER BY city",
                               sqlp.select_from(Customers.City, distinct=True)
                               .order_by(Customers.City))

    def test_select_distinct_2(self):
        self.assertSqlExeEqual("SELECT DISTINCT city, country FROM customers ORDER BY country",
                               sqlp.select_from(Customers.City, Customers.Country, distinct=True)
                               .order_by(Customers.Country))

    def test_select_distinct_null(self):
        self.assertSqlExeEqual("SELECT DISTINCT company FROM customers ORDER BY company nulls first",
                               sqlp.select_from(Customers.Company, distinct=True)
                               .order_by(sqlp.nulls_first(Customers.Company)))


class OrderByTest(SqlTestCase):
    """
    https://www.sqlitetutorial.net/sqlite-order-by/
    """

    def test_select(self):
        self.assertSqlExeEqual("SELECT name, milliseconds, albumid FROM tracks;",
                               sqlp.select_from(Tracks.Name, Tracks.Milliseconds, Tracks.AlbumId))

    def test_ordered(self):
        self.assertSqlExeEqual("""
        SELECT
            name,
            milliseconds,
            albumid
        FROM
            tracks
        ORDER BY
            albumid ASC;
        """, sqlp.select_from(Tracks.Name, Tracks.Milliseconds, Tracks.AlbumId)
                               .order_by(sqlp.asc(Tracks.AlbumId)))

    def test_ordered_two_columns(self):
        self.assertSqlExeEqual("""
         SELECT
            name,
            milliseconds,
            albumid
        FROM
            tracks
        ORDER BY
            albumid ASC,
            milliseconds DESC;
        """, sqlp.select_from(Tracks.Name, Tracks.Milliseconds, Tracks.AlbumId)
                               .order_by(sqlp.asc(Tracks.AlbumId),
                                         sqlp.desc(Tracks.Milliseconds)))

    def test_ordered_by_index(self):
        self.assertSqlExeEqual("""
        SELECT
            name,
            milliseconds,
            albumid
        FROM
            tracks
        ORDER BY
             3,2;
        """, sqlp.select_from(Tracks.Name, Tracks.Milliseconds, Tracks.AlbumId).order_by(3, 2))

    def test_order_null_last(self):
        self.assertSqlExeEqual("""
        SELECT
            TrackId,
            Name,
            Composer
        FROM
            tracks
        ORDER BY
            Composer NULLS LAST;
        """, sqlp.select_from(Tracks.TrackId, Tracks.Name, Tracks.Composer)
                               .order_by(sqlp.nulls_last(Tracks.Composer)))


class WhereTest(SqlTestCase):
    """
    https://www.sqlitetutorial.net/sqlite-where/
    """

    def test_simple(self):
        self.assertSqlExeEqual("""
        SELECT
           name,
           milliseconds,
           bytes,
           albumid
        FROM
           tracks
        WHERE
           albumid = 1;
        """, sqlp.select_from(Tracks.Name, Tracks.Milliseconds, Tracks.Bytes, Tracks.AlbumId)
                               .where(Tracks.AlbumId == 1))

    def test_and(self):
        self.assertSqlExeEqual("""
        SELECT
           name,
           milliseconds,
           bytes,
           albumid
        FROM
           tracks
        WHERE
           albumid = 1
        AND milliseconds > 250000;
        """, sqlp.select_from(Tracks.Name, Tracks.Milliseconds, Tracks.Bytes, Tracks.AlbumId)
                               .where(Tracks.AlbumId == 1,
                                      Tracks.Milliseconds > 250000))

    def test_like(self):
        self.assertSqlExeEqual("""
        SELECT
           name,
           albumid,
           composer
        FROM
           tracks
        WHERE
           composer LIKE '%Smith%'
        ORDER BY
            albumid;
        """, sqlp.select_from(Tracks.Name, Tracks.AlbumId, Tracks.Composer)
                               .where(sqlp.like(Tracks.Composer, '%Smith%'))
                               .order_by(Tracks.AlbumId))

    def test_in(self):
        self.assertSqlExeEqual("""
        SELECT
           name,
           albumid,
           mediatypeid
        FROM
           tracks
        WHERE
           mediatypeid IN (2, 3)
        """, sqlp.select_from(Tracks.Name, Tracks.AlbumId, Tracks.MediaTypeId)
                               .where(sqlp.contains(Tracks.MediaTypeId, (2, 3))))


class LimitTest(SqlTestCase):
    """
    https://www.sqlitetutorial.net/sqlite-limit/
    """

    def test_limit_value(self):
        self.assertSqlExeEqual("""\
        SELECT
            trackId,
            name
        FROM
            tracks
        LIMIT 10;
        """, sqlp.select_from(Tracks.TrackId, Tracks.Name).limit(10))

    def test_limit_offset(self):
        self.assertSqlExeEqual("""\
        SELECT
            trackId,
            name
        FROM
            tracks
        LIMIT 10 OFFSET 10;
        """, sqlp.select_from(Tracks.TrackId, Tracks.Name).limit(10, 10))

    def test_limit_value_with_ordering(self):
        self.assertSqlExeEqual("""\
        SELECT
            trackId,
            name,
            bytes
        FROM
            tracks
        ORDER BY
            bytes DESC
        LIMIT 10;
        """, sqlp.select_from(Tracks.TrackId, Tracks.Name, Tracks.Bytes)
                               .order_by(sqlp.desc(Tracks.Bytes))
                               .limit(10))

    def test_limit_offset_with_ordering(self):
        self.assertSqlExeEqual("""\
        SELECT
            trackId,
            name,
            milliseconds
        FROM
            tracks
        ORDER BY
            milliseconds DESC
        LIMIT 1 OFFSET 1;
        """, sqlp.select_from(Tracks.TrackId, Tracks.Name, Tracks.Milliseconds)
                               .order_by(sqlp.desc(Tracks.Milliseconds))
                               .limit(1, 1))


class BetweenTest(SqlTestCase):
    """
    https://www.sqlitetutorial.net/sqlite-between/
    """

    def test_between(self):
        self.assertSqlExeEqual("""\
        SELECT
            InvoiceId,
            BillingAddress,
            Total
        FROM
            invoices
        WHERE
            Total BETWEEN 14.91 and 18.86
        ORDER BY
            Total;
        """, sqlp.select_from(Invoices.InvoiceId, Invoices.BillingAddress, Invoices.Total)
                               .where(sqlp.between(Invoices.Total, 14.91, 18.86))
                               .order_by(Invoices.Total))

    def test_not_between(self):
        self.assertSqlExeEqual("""\
        SELECT
            InvoiceId,
            BillingAddress,
            Total
        FROM
            invoices
        WHERE
            Total NOT BETWEEN 1 and 20
        ORDER BY
            Total;
        """, sqlp.select_from(Invoices.InvoiceId, Invoices.BillingAddress, Invoices.Total)
                               .where(sqlp.not_between(Invoices.Total, 1, 20))
                               .order_by(Invoices.Total))

    def test_between_date_literal(self):
        self.assertSqlExeEqual("""\
        SELECT
            InvoiceId,
            BillingAddress,
            InvoiceDate,
            Total
        FROM
            invoices
        WHERE
             InvoiceDate BETWEEN '2010-01-01' AND '2010-01-31'
        ORDER BY
            InvoiceDate;
        """, sqlp.select_from(Invoices.InvoiceId, Invoices.BillingAddress, Invoices.InvoiceDate, Invoices.Total)
                               .where(sqlp.between(Invoices.InvoiceDate, '2010-01-01', '2010-01-31'))
                               .order_by(Invoices.InvoiceDate))

    def test_between_date(self):
        self.assertSqlExeEqual("""\
        SELECT
            InvoiceId,
            BillingAddress,
            InvoiceDate,
            Total
        FROM
            invoices
        WHERE
             InvoiceDate BETWEEN '2010-01-01' AND '2010-01-31'
        ORDER BY
            InvoiceDate;
        """, sqlp.select_from(Invoices.InvoiceId, Invoices.BillingAddress, Invoices.InvoiceDate, Invoices.Total)
                               .where(sqlp.between(Invoices.InvoiceDate,
                                                   datetime.date(2010, 1, 1),
                                                   datetime.date(2010, 1, 31)))
                               .order_by(Invoices.InvoiceDate))

    def test_not_between_date_literal(self):
        self.assertSqlExeEqual("""\
        SELECT
            InvoiceId,
            BillingAddress,
            date(InvoiceDate) InvoiceDate,
            Total
        FROM
            invoices
        WHERE
             InvoiceDate NOT BETWEEN '2009-01-03' AND '2013-12-01'
        ORDER BY
            InvoiceDate;
        """, sqlp.select_from(Invoices.InvoiceId, Invoices.BillingAddress,
                              sqlp.date(Invoices.InvoiceDate) @ 'InvoiceDate',
                              Invoices.Total)
                               .where(sqlp.not_between(Invoices.InvoiceDate, '2009-01-03', '2013-12-01'))
                               .order_by(Invoices.InvoiceDate))

    def test_not_between_date(self):
        self.assertSqlExeEqual("""\
        SELECT
            InvoiceId,
            BillingAddress,
            date(InvoiceDate) InvoiceDate,
            Total
        FROM
            invoices
        WHERE
             InvoiceDate NOT BETWEEN '2009-01-03' AND '2013-12-01'
        ORDER BY
            InvoiceDate
        LIMIT 100;
        """, sqlp.select_from(Invoices.InvoiceId, Invoices.BillingAddress,
                              sqlp.date(Invoices.InvoiceDate) @ 'InvoiceDate',
                              Invoices.Total)
                               .where(sqlp.not_between(Invoices.InvoiceDate,
                                                       datetime.date(2009, 1, 3),
                                                       datetime.date(2013, 12, 1)))
                               .order_by(Invoices.InvoiceDate)
                               .limit(100))


class InTest(SqlTestCase):
    """
    https://www.sqlitetutorial.net/sqlite-in/
    """

    def test_in_list(self):
        self.assertSqlExeEqual("""\
        SELECT
            TrackId,
            Name,
            Mediatypeid
        FROM
            Tracks
        WHERE
            MediaTypeId IN (1, 2)
        ORDER BY
            Name ASC;
        """, sqlp.select_from(Tracks.TrackId, Tracks.Name, Tracks.MediaTypeId)
                               .where(sqlp.contains(Tracks.MediaTypeId, (1, 2)))
                               .order_by(sqlp.asc(Tracks.Name)))

    def test_in_list_by_or(self):
        self.assertSqlExeEqual("""\
        SELECT
            TrackId,
            Name,
            Mediatypeid
        FROM
            Tracks
        WHERE
            MediaTypeId = 1 OR MediaTypeId = 2
        ORDER BY
            Name ASC;
        """, sqlp.select_from(Tracks.TrackId, Tracks.Name, Tracks.MediaTypeId)
                               .where((Tracks.MediaTypeId == 1) | (Tracks.MediaTypeId == 2))
                               .order_by(sqlp.asc(Tracks.Name)))

    def test_in_sub_query(self):
        self.assertSqlExeEqual("""\
        SELECT
            TrackId,
            Name,
            AlbumId
        FROM
            Tracks
        WHERE
            AlbumId IN (
                SELECT
                    AlbumId
                FROM
                    Albums
                WHERE
                    ArtistId = 12
            );
        """, sqlp.select_from(Tracks.TrackId, Tracks.Name, Tracks.AlbumId)
                               .where(sqlp.contains(Tracks.AlbumId, sqlp.select_from(Albums.AlbumId)
                                                    .where(Albums.ArtistId == 12))))

    def test_not_in(self):
        self.assertSqlExeEqual("""\
        SELECT
            trackid,
            name,
            genreid
        FROM
            tracks
        WHERE
            genreid NOT IN (1, 2,3)
        LIMIT 100;
        """, sqlp.select_from(Tracks.TrackId, Tracks.Name, Tracks.GenreId)
                               .where(sqlp.not_contains(Tracks.GenreId, (1, 2, 3)))
                               .limit(100))


class LikeGlobTest(SqlTestCase):
    """
    https://www.sqlitetutorial.net/sqlite-like/
    https://www.sqlitetutorial.net/sqlite-glob/
    """

    def test_like(self):
        self.assertSqlExeEqual("""\
        SELECT
            trackid,
            name
        FROM
            tracks
        WHERE
            name LIKE 'Wild%'
        """, sqlp.select_from(Tracks.TrackId, Tracks.Name)
                               .where(sqlp.like(Tracks.Name, 'Wild%')))

        self.assertSqlExeEqual("""\
        SELECT
            trackid,
            name
        FROM
            tracks
        WHERE
            name LIKE '%Wild'
        """, sqlp.select_from(Tracks.TrackId, Tracks.Name)
                               .where(sqlp.like(Tracks.Name, '%Wild')))

        self.assertSqlExeEqual("""\
        SELECT
            trackid,
            name
        FROM
            tracks
        WHERE
            name LIKE '%Wild%'
        """, sqlp.select_from(Tracks.TrackId, Tracks.Name)
                               .where(sqlp.like(Tracks.Name, '%Wild%')))

        self.assertSqlExeEqual("""\
        SELECT
            trackid,
            name
        FROM
            tracks
        WHERE
            name LIKE '%Br_wn%'
        """, sqlp.select_from(Tracks.TrackId, Tracks.Name)
                               .where(sqlp.like(Tracks.Name, '%Br_wn%')))

    def test_glob(self):
        self.assertSqlExeEqual("""\
        SELECT
            trackid,
            name
        FROM
            tracks
        WHERE
            name GLOB 'Man*';
        """, sqlp.select_from(Tracks.TrackId, Tracks.Name)
                               .where(sqlp.glob(Tracks.Name, 'Man*')))

        self.assertSqlExeEqual("""\
        SELECT
            trackid,
            name
        FROM
            tracks
        WHERE
            name GLOB '*Man';
        """, sqlp.select_from(Tracks.TrackId, Tracks.Name)
                               .where(sqlp.glob(Tracks.Name, '*Man')))

        self.assertSqlExeEqual("""\
        SELECT
            trackid,
            name
        FROM
            tracks
        WHERE
            name GLOB '?ere*';
        """, sqlp.select_from(Tracks.TrackId, Tracks.Name)
                               .where(sqlp.glob(Tracks.Name, '?ere*')))

        self.assertSqlExeEqual("""\
        SELECT
            trackid,
            name
        FROM
            tracks
        WHERE
            name GLOB '*[1-9]*';
        """, sqlp.select_from(Tracks.TrackId, Tracks.Name)
                               .where(sqlp.glob(Tracks.Name, '*[1-9]*')))

        self.assertSqlExeEqual("""\
        SELECT
            trackid,
            name
        FROM
            tracks
        WHERE
            name GLOB '*[^1-9]*';
        """, sqlp.select_from(Tracks.TrackId, Tracks.Name)
                               .where(sqlp.glob(Tracks.Name, '*[^1-9]*')))


class IsNullTest(SqlTestCase):
    """
    https://www.sqlitetutorial.net/sqlite-is-null/
    """

    def test_eq_null(self):
        self.assertSqlExeEqual("""\
        SELECT
            Name,
            Composer
        FROM
            tracks
        WHERE
            Composer = NULL;
        """, sqlp.select_from(Tracks.Name, Tracks.Composer)
                               .where(Tracks.Composer == None))

    def test_is_null(self):
        self.assertSqlExeEqual("""\
        SELECT
            Name,
            Composer
        FROM
            tracks
        WHERE
            Composer IS NULL
        ORDER BY
            Name;
        """, sqlp.select_from(Tracks.Name, Tracks.Composer)
                               .where(sqlp.is_null(Tracks.Composer))
                               .order_by(Tracks.Name))

    def test_is_not_null(self):
        self.assertSqlExeEqual("""\
        SELECT
            Name,
            Composer
        FROM
            tracks
        WHERE
            Composer IS NOT NULL
        ORDER BY
            Name;
        """, sqlp.select_from(Tracks.Name, Tracks.Composer)
                               .where(sqlp.is_not_null(Tracks.Composer))
                               .order_by(Tracks.Name))


class CteTest(SqlTestCase):
    """https://www.sqlitetutorial.net/sqlite-cte/"""

    @unittest.skip('not implemented yet')
    def test_simple_case(self):
        self.assertSqlExeEqual("""\
        WITH top_tracks AS (
            SELECT trackid, name
            FROM tracks
            ORDER BY trackid
            LIMIT 5
        )
        SELECT * FROM top_tracks;
        """, None)

    @unittest.skip('not implemented yet')
    def test_complex_case(self):
        self.assertSqlExeEqual("""\
        WITH customer_sales AS (
            SELECT c.customerid,
                   c.firstname || ' ' || c.lastname AS customer_name,
                   ROUND(SUM(ii.unitprice * ii.quantity),2) AS total_sales
            FROM customers c
            INNER JOIN invoices i ON c.customerid = i.customerid
            INNER JOIN invoice_items ii ON i.invoiceid = ii.invoiceid
            GROUP BY c.customerid
        )
        SELECT customer_name, total_sales 
        FROM customer_sales
        ORDER BY total_sales DESC, customer_name
        LIMIT 5;
        """, None)

class CaseTest(SqlTestCase):
    """https://www.sqlitetutorial.net/sqlite-case/"""

    def test_literal_case(self):
        self.assertSqlExeEqual("""\
        SELECT customerid,
               firstname,
               lastname,
               CASE country 
                   WHEN 'USA' 
                       THEN 'Domestic' 
                   ELSE 'Foreign' 
               END CustomerGroup
        FROM 
            customers
        ORDER BY 
            LastName,
            FirstName;
        """, sqlp.select_from(
            Customers.CustomerId,
            Customers.FirstName,
            Customers.LastName,
            (sqlp.case(Customers.Country)
             .when('USA', 'Domestic')
             .else_('Foreign')) @ 'CustomerGroup'
        ).order_by(Customers.LastName, Customers.FirstName))

    def test_condition_case(self):
        self.assertSqlExeEqual("""\
        SELECT
            trackid,
            name,
            CASE
                WHEN milliseconds < 60000                           
                THEN 'short'
                WHEN milliseconds > 60000 AND milliseconds < 300000 
                THEN 'medium'
                ELSE 'long'
            END category
        FROM
            tracks;
        """, sqlp.select_from(
            Tracks.TrackId,
            Tracks.Name,
            (sqlp.case()
             .when(Tracks.Milliseconds < 60000, 'short')
             .when((Tracks.Milliseconds > 60000) & (Tracks.Milliseconds < 30_0000), 'medium')
             .else_('long')) @ 'category'
        ))

if __name__ == '__main__':
    unittest.main()
