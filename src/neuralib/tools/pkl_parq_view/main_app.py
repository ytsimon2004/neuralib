from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from rich.pretty import pprint

from neuralib.util.util_type import PathLike


class PickleParquetViewer:

    def __init__(self, file: PathLike,
                 view_array_shape: bool = False,
                 view_iterable_len: bool = False):
        """
        Simple CLI Viewer for customized pickle/parquet file extension

        :param file: file path
        :param view_array_shape: if field as an array type, only view its shape
        :param view_iterable_len: if field as an iterable object (dict, set, list, tuple), only view its length
        """
        self.file = file
        self.suffix = Path(file).suffix

        self.view_shape_only = view_array_shape
        self.view_iterable_len = view_iterable_len

    def view(self):
        if self.suffix in ('.pkl', '.pickle'):
            self._view_pkl()
        elif self.suffix == '.parquet':
            self._view_parquet()
        else:
            raise ValueError(f'{self.suffix} not support')

    def _view_pkl(self) -> None:
        import pickle
        with open(self.file, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            if not self.view_shape_only and not self.view_iterable_len:
                pprint(data)
            else:
                for key, val in data.items():
                    if hasattr(val, 'shape') and self.view_shape_only:
                        info = val.shape
                    elif hasattr(val, '__len__') and self.view_iterable_len:
                        info = len(val)
                    else:
                        info = val
                    pprint(f'{key}: {info}')

        elif isinstance(data, pd.DataFrame):
            data = pd.read_pickle(self.file)
            if self.view_shape_only:
                data = data.applymap(lambda x: x.shape if isinstance(x, np.ndarray) else x)

            if self.view_iterable_len:
                data = data.applymap(lambda x: len(x) if isinstance(x, (list, tuple, set, dict)) else x)

            pprint(data)

        else:
            raise TypeError(f'not supported {type(data)} pickle file')

    def _view_parquet(self):
        from neuralib.util.util_verbose import printdf
        data = pl.read_parquet(self.file)

        # only consider list[Any] column type in polars dataframe
        if self.view_iterable_len:
            for s in data.iter_columns():
                if isinstance(s.dtype, pl.List):
                    data = data.with_columns(pl.col(s.name).list.len())

        printdf(data)


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Simple CLI Viewer for customized pickle/parquet file extension')

    ap.add_argument(metavar='FILE', help='pickle/parquet file path', dest='file')
    ap.add_argument('-S', '--shape', action='store_true', help='if field as an array type, only view its shape')
    ap.add_argument('-L', '--len', action='store_true',
                    help='if field as an iterable object (dict, set, list, tuple), only view its length')

    opt = ap.parse_args()

    PickleParquetViewer(opt.file, view_array_shape=opt.shape, view_iterable_len=opt.len).view()


if __name__ == '__main__':
    main()
