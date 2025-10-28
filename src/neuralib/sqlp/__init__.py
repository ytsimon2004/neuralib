import warnings

warnings.warn(
    "The 'sqlp' module is deprecated and will be removed in a separated module. "
    "Use 'pip install sqlclz' instead.'",
    DeprecationWarning,
    stacklevel=2
)

from . import util
from .annotation import *
from .cli import *
from .connection import *
from .func import *
from .func_date import *
from .func_stat import *
from .func_win import *
from .literal import *
from .stat import Cursor
from .stat_start import *
from .table import *
from .table_nt import *
