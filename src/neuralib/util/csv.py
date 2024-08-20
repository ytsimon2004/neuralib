from __future__ import annotations

from neuralib.util.deprecation import deprecated_func

__all__ = ['csv_header']


@deprecated_func(new='neuralib.io.csv_header()', removal_version='0.3.0')
def csv_header(*args, **kwargs):
    from neuralib.io import csv_header
    return csv_header(*args, **kwargs)
