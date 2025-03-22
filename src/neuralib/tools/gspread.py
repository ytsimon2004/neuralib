from pathlib import Path
from typing import Union, Any, get_args, Final, Literal

import gspread
import numpy as np
import pandas as pd
import polars as pl
from neuralib.typing import PathLike, DataFrame
from neuralib.util.verbose import fprint
from typing_extensions import TypeAlias, Self

__all__ = [
    'SpreadSheetName',
    'WorkPageName',
    'DataIndex',
    #
    'GoogleSpreadSheet',
    'GoogleWorkSheet',
    #
    'upload_dataframe_to_spreadsheet'
]

SpreadSheetName: TypeAlias = str
"""spreadsheet name"""

WorkPageName: TypeAlias = str
"""workpage name of the spreadsheet"""

DataIndex: TypeAlias = Union[None, int, str, slice, list[int], list[str], np.ndarray]
"""data index type"""

VALUE_RENDER_OPT = Literal['FORMATTED_VALUE', 'UNFORMATTED_VALUE', 'FORMULA']
"""value render option for the cell"""


class GoogleWorkSheet:

    def __init__(self, worksheet: gspread.Worksheet,
                 primary_key: str | tuple[str, ...] = 'Data'):
        """

        :param worksheet: ``gspread.Worksheet``
        :param primary_key: Primary key of the worksheet.

            If str type, it must be one of the column name.

            If tuple str type, the primary key is join using "_" per row
        """
        self._worksheet = worksheet
        self._headers = tuple(self._worksheet.row_values(1))

        if isinstance(primary_key, str) and primary_key not in self._headers:
            raise ValueError(f'col not found: {primary_key}')
        elif isinstance(primary_key, (str, tuple)):
            self.primary_key: Final[str, tuple[str, ...]] = primary_key
        else:
            raise TypeError(f'{primary_key}')

    @classmethod
    def of(cls,
           name: SpreadSheetName,
           page: WorkPageName,
           service_account_path: PathLike,
           primary_key: str | tuple[str, ...] = 'Data') -> Self:
        """
        Get a worksheet from spreadsheet

        :param name: ``SpreadSheetName``
        :param page: ``WorkPageName``
        :param service_account_path: The path to the service account json file
        :param primary_key: Primary key of the worksheet.

            If str type, it must be one of the column name.

            If tuple str type, the primary key is join using "_" per row

        :return: ``GoogleWorkSheet``
        """
        sh = GoogleSpreadSheet(name, service_account_path, primary_key)

        if page not in sh:
            raise ValueError(f'{page} not found in spreadsheet: {name}')

        return sh.get_worksheet(page)

    @property
    def title(self) -> WorkPageName:
        """``WorkPageName``"""
        return self._worksheet.title

    @property
    def headers(self) -> list[str]:
        """list of worksheet header"""
        return list(self._headers)

    @property
    def primary_key_list(self) -> list[str]:
        """list of primary key of the worksheet"""
        primary = self.primary_key
        if isinstance(primary, str):
            return self.values(primary)
        elif isinstance(primary, tuple):
            ks = [self.values(p) for p in primary]

            if len(set(list(map(len, ks)))) != 1:
                print(set(list(map(len, ks))))
                raise RuntimeError(f'primary key cannot join properly due to different len in col: {self.primary_key}')

            return [
                '_'.join([str(it) for it in j])
                for j in (list(zip(*ks)))
            ]
        else:
            raise TypeError(f'{self.primary_key}')

    def get_range_value(self, a1_range_notation: str) -> list[Any]:
        """get values from range notation. i.e., `B1:S1` to get the list of content.

        If get values from 2D, the return order would be first column-wise, then row-wise

        """
        range_values = self._worksheet.range(a1_range_notation)
        return [it.value for it in range_values]

    def values(self, head: str) -> list[Any]:
        """get list of value from header"""
        col = self._col(head)
        return list(self._worksheet.col_values(col)[1:])

    def _row(self, data: DataIndex) -> Union[None, int, list[int], np.ndarray]:
        """
        Get row(s) index

        :param data: *str: first col value; *int: index
        :return:
        """
        if data is None:
            return None
        if isinstance(data, int):
            return data + 2  # skip header row + one-base
        elif isinstance(data, str):
            return self.primary_key_list.index(data) + 2
        elif isinstance(data, list):
            if len(data) == 0:
                return []
            return [self._row(it) for it in data]
        elif isinstance(data, (slice, np.ndarray)):
            return np.arange(len(self.primary_key_list))[data] + 2

        raise TypeError()

    def _col(self, head: str) -> int:
        """Get column index based on header (one-base)"""
        return self._headers.index(head) + 1

    def _rowcol(self, data: DataIndex, head: str) -> tuple[None | int | list[int], int]:
        """Get row in col indices"""
        return self._row(data), self._col(head)

    # noinspection PyTypeChecker
    def get_cell(self,
                 data: DataIndex,
                 head: str,
                 value_render_option: VALUE_RENDER_OPT = 'FORMATTED_VALUE'):
        """
        Get data from a cell

        :param data: ``DataIndex``
        :param head: header name
        :param value_render_option: ``VALUE_RENDER_OPT``: {'FORMATTED_VALUE', 'UNFORMATTED_VALUE', 'FORMULA'}
        :return:
        """
        if value_render_option not in get_args(VALUE_RENDER_OPT):
            raise ValueError('')

        row, col = self._rowcol(data, head)

        if row is None:
            return self._worksheet.col_values(col, value_render_option=value_render_option)[1:]
        elif isinstance(row, int):
            return self._worksheet.cell(row, col, value_render_option=value_render_option).value
        else:
            return [self._worksheet.cell(it, col, value_render_option=value_render_option).value for it in row]

    def update_cell(self, data: DataIndex,
                    head: str,
                    value: list[str] | str):
        """
        Update value in a cell

        :param data: ``DataIndex``
        :param head: header name
        :param value: value to be updated. str type if single field
        :return:
        """
        from gspread.utils import rowcol_to_a1

        row, col = self._rowcol(data, head)
        if row is None:
            if len(value) != len(self.primary_key_list):
                raise ValueError()

            data = []
            for it, v in enumerate(value):
                data.append({'range': rowcol_to_a1(it + 2, col), 'values': [[v]]})
                fprint(f'UPDATES: {rowcol_to_a1(it + 2, col)} from {self.get_cell(it, head)} -> {v}', vtype='io')

            self._worksheet.batch_update(data)

        elif isinstance(row, int):
            self._worksheet.update_cell(row, col, value)

        else:
            if len(value) != len(row):
                raise ValueError()

            data = []
            for it, v in zip(row, value):
                data.append({'range': rowcol_to_a1(it, col), 'values': [[v]]})
                fprint(f'UPDATES: {rowcol_to_a1(it, col)} from {self.get_cell(it - 2, head)}-> {v}', vtype='io')

            self._worksheet.batch_update(data)

    def clear(self):
        """Clears all cells in the worksheet"""
        self._worksheet.clear()

    def update(self, range_name, values=None, **kwargs):
        """Sets values in a cell range of the sheet"""
        self._worksheet.update(range_name, values, **kwargs)

    def to_pandas(self) -> pd.DataFrame:
        """Worksheet to pandas dataframe"""
        return pd.DataFrame(self._worksheet.get_all_records())

    def to_polars(self) -> pl.DataFrame:
        """Worksheet to polar dataframe"""
        return pl.DataFrame(self._worksheet.get_all_records(), nan_to_null=True)


# =========== #
# Spreadsheet #
# =========== #

class GoogleSpreadSheet:
    """
    Gspread module wrapper to access `google spreadsheet`

    .. seealso:: `<https://docs.gspread.org/en/latest/>`_


    .. code-block:: python

        service_account_path = ...  # The path to the service account json file
        sh = GoogleSpreadSheet('test_sheet', service_account_path)

        # get a worksheet from the spreadsheet
        ws = sh[WORK_SHEET_NAME]

    """

    def __init__(self, name: SpreadSheetName,
                 service_account_path: PathLike,
                 primary_key: str | tuple[str, ...] = 'Data'):
        """

        :param name: name of the spreadsheet
        :param service_account_path: The path to the service account json file
        :param primary_key: Primary key of the worksheet.

            If str type, it must be one of the column name.

            If tuple str type, the primary key is join using "_" per row
        """

        self._client = gspread.service_account(filename=service_account_path)

        self._sheet: gspread.Spreadsheet = self._client.open(name)
        self._worksheets: list[gspread.Worksheet] = self._sheet.worksheets()
        self._primary_key = primary_key

    def __len__(self):
        """number of worksheet"""
        return len(self._worksheets)

    def __contains__(self, item: WorkPageName) -> bool:
        return self.has_worksheet(item)

    def __getitem__(self, item: int | WorkPageName) -> GoogleWorkSheet:
        """get the worksheet.

        if int type, get worksheet from index.

        if str type, get worksheet from ``WorkPageName``

        """
        if isinstance(item, int):
            return GoogleWorkSheet(self._worksheets[item])
        elif isinstance(item, str):
            try:
                return self.get_worksheet(item)
            except ValueError:
                pass

            raise KeyError(f'{item} not found')
        else:
            raise TypeError(f'invalid item type: {type(item)}')

    @property
    def title(self) -> SpreadSheetName:
        """``SpreadSheetName``"""
        return self._sheet.title

    @property
    def worksheet_list(self) -> list[WorkPageName]:
        """list of ``WorkPageName``"""
        return [it.title for it in self._worksheets]

    def has_worksheet(self, page: WorkPageName) -> bool:
        """If has the worksheet, implement also in ``__contains__()``"""
        for w in self._worksheets:
            if w.title == page:
                return True
        return False

    def get_worksheet(self, page: WorkPageName) -> GoogleWorkSheet:
        """Get the worksheet. implement also in ``__get_item__()``"""
        for w in self._worksheets:
            if w.title == page:
                return GoogleWorkSheet(w, self._primary_key)

        raise ValueError(f'page not found: {page}')


# ========= #
# Utilities #
# ========= #

def upload_dataframe_to_spreadsheet(df: DataFrame,
                                    gspread_name: SpreadSheetName,
                                    worksheet_name: WorkPageName,
                                    service_account_path: Path | None,
                                    primary_key: str | tuple[str, ...] = 'Data') -> None:
    """
    Upload a dataframe to a gspread worksheet

    :param df: polars or pandas DataFrame
    :param gspread_name: spreadsheet name
    :param worksheet_name: worksheet name under the spreadsheet
    :param service_account_path: The path to the service account json file.
    :param primary_key: Primary key of the worksheet.
            If str type, it must be one of the column name.
            If tuple str type, the primary key is join using "_" per row
    """
    gs = GoogleSpreadSheet(gspread_name, service_account_path, primary_key)
    spreadsheet = gs._sheet

    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    if worksheet_name not in gs.worksheet_list:
        spreadsheet.add_worksheet(title=worksheet_name, rows=df.shape[0], cols=len(df.columns))
        fprint(f'ADD WORKSHEET: {worksheet_name}')
        gs = GoogleSpreadSheet(gspread_name, service_account_path, primary_key)  # refresh page

    worksheet = gs[worksheet_name]
    worksheet.clear()

    # cast dtype avoid serialization problem from gspread
    df = df.cast({pl.Datetime: pl.Utf8})

    data = [df.columns] + [field for field in df.iter_rows()]
    worksheet.update(data)
