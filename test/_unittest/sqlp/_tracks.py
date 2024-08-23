"""
https://www.sqlitetutorial.net/sqlite-sample-database/

wget https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip; unzip chinook.zip
"""
from __future__ import annotations

import datetime
from typing import NamedTuple, Annotated, Optional

from neuralib.sqlp import named_tuple_table_class, foreign, PRIMARY

__all__ = ['Artists', 'Genres', 'MediaTypes', 'Albums', 'Tracks', 'Employees', 'Customers', 'Invoices', 'Invoice_Items']


@named_tuple_table_class
class Artists(NamedTuple):
    """
    CREATE TABLE "artists" (
        [ArtistId] INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        [Name] NVARCHAR(120)
    )

    """
    ArtistId: Annotated[int, PRIMARY(auto_increment=True)]
    Name: Optional[str]


@named_tuple_table_class
class Genres(NamedTuple):
    """
    CREATE TABLE "genres" (
        [GenreId] INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        [Name] NVARCHAR(120)
    )
    """
    GenreId: Annotated[int, PRIMARY(auto_increment=True)]
    Name: Optional[str]


@named_tuple_table_class
class MediaTypes(NamedTuple):
    """
    CREATE TABLE "media_types" (
        [MediaTypeId] INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        [Name] NVARCHAR(120)
    )

    """
    MediaTypeId: Annotated[int, PRIMARY(auto_increment=True)]
    Name: Optional[str]


@named_tuple_table_class
class Albums(NamedTuple):
    """
    CREATE TABLE "albums" (
        [AlbumId] INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        [Title] NVARCHAR(160)  NOT NULL,
        [ArtistId] INTEGER  NOT NULL,
        FOREIGN KEY ([ArtistId]) REFERENCES "artists" ([ArtistId])
            ON DELETE NO ACTION ON UPDATE NO ACTION
    )

    """
    AlbumId: Annotated[int, PRIMARY(auto_increment=True)]
    Title: str
    ArtistId: int

    @foreign(Artists)
    def _artists(self):
        return self.ArtistId


@named_tuple_table_class
class Tracks(NamedTuple):
    """
    CREATE TABLE "tracks" (
        [TrackId] INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        [Name] NVARCHAR(200)  NOT NULL,
        [AlbumId] INTEGER,
        [MediaTypeId] INTEGER  NOT NULL,
        [GenreId] INTEGER,
        [Composer] NVARCHAR(220),
        [Milliseconds] INTEGER  NOT NULL,
        [Bytes] INTEGER,
        [UnitPrice] NUMERIC(10,2)  NOT NULL,
        FOREIGN KEY ([AlbumId]) REFERENCES "albums" ([AlbumId])
            ON DELETE NO ACTION ON UPDATE NO ACTION,
        FOREIGN KEY ([GenreId]) REFERENCES "genres" ([GenreId])
            ON DELETE NO ACTION ON UPDATE NO ACTION,
        FOREIGN KEY ([MediaTypeId]) REFERENCES "media_types" ([MediaTypeId])
            ON DELETE NO ACTION ON UPDATE NO ACTION
    )

    """

    TrackId: Annotated[int, PRIMARY(auto_increment=True)]
    Name: str
    AlbumId: Optional[int]
    MediaTypeId: int
    GenreId: Optional[int]
    Composer: Optional[str]
    Milliseconds: int
    Bytes: Optional[int]
    UnitPrice: float

    @foreign(Albums)
    def _albums(self):
        return self.AlbumId

    @foreign(Genres)
    def _genres(self):
        return self.GenreId

    @foreign(MediaTypes)
    def _media_types(self):
        return self.MediaTypeId


@named_tuple_table_class
class Employees(NamedTuple):
    """
    CREATE TABLE "employees" (
        [EmployeeId] INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        [LastName] NVARCHAR(20)  NOT NULL,
        [FirstName] NVARCHAR(20)  NOT NULL,
        [Title] NVARCHAR(30),
        [ReportsTo] INTEGER,
        [BirthDate] DATETIME,
        [HireDate] DATETIME,
        [Address] NVARCHAR(70),
        [City] NVARCHAR(40),
        [State] NVARCHAR(40),
        [Country] NVARCHAR(40),
        [PostalCode] NVARCHAR(10),
        [Phone] NVARCHAR(24),
        [Fax] NVARCHAR(24),
        [Email] NVARCHAR(60),
        FOREIGN KEY ([ReportsTo]) REFERENCES "employees" ([EmployeeId])
                    ON DELETE NO ACTION ON UPDATE NO ACTION
    )

    """
    EmployeeId: Annotated[int, PRIMARY(auto_increment=True)]
    LastName: str
    FirstName: str
    Title: Optional[str]
    ReportsTo: Optional[int]
    BirthDate: Optional[datetime.datetime]
    HireDate: Optional[datetime.datetime]
    Address: Optional[str]
    City: Optional[str]
    State: Optional[str]
    Country: Optional[str]
    PostalCode: Optional[str]
    Phone: Optional[str]
    Fax: Optional[str]
    Email: Optional[str]

    @foreign('EmployeeId')
    def _report_to(self):
        return self.ReportsTo


@named_tuple_table_class
class Customers(NamedTuple):
    """
    CREATE TABLE "customers" (
        [CustomerId] INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        [FirstName] NVARCHAR(40)  NOT NULL,
        [LastName] NVARCHAR(20)  NOT NULL,
        [Company] NVARCHAR(80),
        [Address] NVARCHAR(70),
        [City] NVARCHAR(40),
        [State] NVARCHAR(40),
        [Country] NVARCHAR(40),
        [PostalCode] NVARCHAR(10),
        [Phone] NVARCHAR(24),
        [Fax] NVARCHAR(24),
        [Email] NVARCHAR(60)  NOT NULL,
        [SupportRepId] INTEGER,
        FOREIGN KEY ([SupportRepId]) REFERENCES "employees" ([EmployeeId])
                    ON DELETE NO ACTION ON UPDATE NO ACTION
    )

    """
    CustomerId: Annotated[int, PRIMARY(auto_increment=True)]
    FirstName: str
    LastName: str
    Company: Optional[str]
    Address: Optional[str]
    City: Optional[str]
    State: Optional[str]
    Country: Optional[str]
    PostalCode: Optional[str]
    Phone: Optional[str]
    Fax: Optional[str]
    Email: str
    SupportRepId: Optional[int]

    @foreign(Employees)
    def _support_res(self):
        return self.SupportRepId


@named_tuple_table_class
class Invoices(NamedTuple):
    """
    CREATE TABLE "invoices" (
        [InvoiceId] INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        [CustomerId] INTEGER  NOT NULL,
        [InvoiceDate] DATETIME  NOT NULL,
        [BillingAddress] NVARCHAR(70),
        [BillingCity] NVARCHAR(40),
        [BillingState] NVARCHAR(40),
        [BillingCountry] NVARCHAR(40),
        [BillingPostalCode] NVARCHAR(10),
        [Total] NUMERIC(10,2)  NOT NULL,
        FOREIGN KEY ([CustomerId]) REFERENCES "customers" ([CustomerId])
                    ON DELETE NO ACTION ON UPDATE NO ACTION
    )

    """
    InvoiceId: Annotated[int, PRIMARY(auto_increment=True)]
    CustomerId: int
    InvoiceDate: datetime.datetime
    BillingAddress: Optional[str]
    BillingCity: Optional[str]
    BillingState: Optional[str]
    BillingCountry: Optional[str]
    BillingPostalCode: Optional[str]
    Total: int

    @foreign(Customers)
    def _customer(self):
        return self.CustomerId


@named_tuple_table_class
class Invoice_Items(NamedTuple):
    """
    CREATE TABLE "invoice_items" (
        [InvoiceLineId] INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        [InvoiceId] INTEGER  NOT NULL,
        [TrackId] INTEGER  NOT NULL,
        [UnitPrice] NUMERIC(10,2)  NOT NULL,
        [Quantity] INTEGER  NOT NULL,
        FOREIGN KEY ([InvoiceId]) REFERENCES "invoices" ([InvoiceId])
            ON DELETE NO ACTION ON UPDATE NO ACTION,
        FOREIGN KEY ([TrackId]) REFERENCES "tracks" ([TrackId])
            ON DELETE NO ACTION ON UPDATE NO ACTION
    )
    """

    InvoiceLineId: Annotated[int, PRIMARY(auto_increment=True)]
    InvoiceId: int
    TrackId: int
    UnitPrice: float
    Quantity: int

    @foreign(Invoices)
    def _invoices(self):
        return self.InvoiceId

    @foreign(Tracks)
    def _tracks(self):
        return self.TrackId
