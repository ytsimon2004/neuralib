"""
SQLp
====

Incubator module
----------------

It is an incubator module.

"""
from . import util, alter
from .cli import Database
from .connection import *
from .expr import SqlExpr
from .func import *
from .func_date import *
from .func_stat import *
from .func_win import *
from .literal import *
from .stat import create_table, select_from, insert_into, update, delete_from, Cursor
from .table import foreign, check, PRIMARY, UNIQUE
from .table_nt import *
