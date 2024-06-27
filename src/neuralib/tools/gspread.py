from __future__ import annotations

from typing import Union, Any, get_args

import gspread
import numpy as np
import pandas as pd
import polars as pl
from typing_extensions import TypeAlias, Self, Literal

from neuralib.util.util_type import PathLike, DataFrame
from neuralib.util.util_verbose import fprint

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
WorkPageName: TypeAlias = str
DataIndex: TypeAlias = Union[None, int, str, slice, list[int], list[str], np.ndarray]

VALUE_RENDER_OPT = Literal['FORMATTED_VALUE', 'UNFORMATTED_VALUE', 'FORMULA']


class GoogleWorkSheet:

    def __init__(self, worksheet: gspread.Worksheet,
                 index_col_name: str = 'Data'):
        """

        :param worksheet:
        :param index_col_name: index column name (field for the table index. i.e., column for animal/experimental ID)
        """
        self._worksheet = worksheet
        self._headers = tuple(self._worksheet.row_values(1))

        if index_col_name not in self._headers:
            raise ValueError(f'col not found: {index_col_name}')
        self._index_col_name = index_col_name

    @classmethod
    def of(cls,
           name: SpreadSheetName,
           page: WorkPageName,
           service_account_path: PathLike,
           index_col_name: str = 'Data') -> Self:
        """
        Get a worksheet from spreadsheet

        :param name: ``SpreadSheetName``
        :param page: ``WorkPageName``
        :param service_account_path: The path to the service account json file
        :param index_col_name: index column name (field for the table index. i.e., column for animal/experimental ID)
        :return:
        """
        sh = GoogleSpreadSheet(name, service_account_path)

        if page not in sh:
            raise ValueError(f'{page} not found in spreadsheet: {name}')

        return sh.get_worksheet(page, index_col_name)

    @property
    def title(self) -> WorkPageName:
        """``WorkPageName``"""
        return self._worksheet.title

    @property
    def headers(self) -> list[str]:
        """list of worksheet header"""
        return list(self._headers)

    @property
    def index_value(self) -> list[str]:
        """index_value of the worksheet (i.e., primary key)"""
        return self.values(self._index_col_name)

    def get_range_value(self, a1_range_notation: str) -> list[Any]:
        """get values from range notation. i.e., `B1:S1` to get the list of content.

        If get values from 2D, the return order would be first column-wise, then row-wise

        """
        range_values = self._worksheet.range(a1_range_notation)
        return [it.value for it in range_values]

    def values(self, head: str) -> list[Any]:
        col = self._col(head)
        return list(self._worksheet.col_values(col)[1:])

    def _row(self, data: DataIndex) -> Union[None, int, list[int], np.ndarray]:
        """
        get row(s) index

        :param data: *str: first col value; *int: index
        :return:
        """
        if data is None:
            return None
        if isinstance(data, int):
            return data + 2  # skip header row + one-base
        elif isinstance(data, str):
            return self.index_value.index(data) + 2
        elif isinstance(data, list):
            if len(data) == 0:
                return []
            return [self._row(it) for it in data]
        elif isinstance(data, (slice, np.ndarray)):
            return np.arange(len(self.index_value))[data] + 2

        raise TypeError()

    def _col(self, head: str) -> int:
        """get column index based on header (one-base)"""
        return self._headers.index(head) + 1

    def _rowcol(self, data: DataIndex, head: str) -> tuple[None | int | list[int], int]:
        """get row in col indices"""
        return self._row(data), self._col(head)

    def get_cell(self, data: DataIndex,
                 head: str,
                 value_render_option: VALUE_RENDER_OPT = 'FORMATTED_VALUE'):
        # noinspection PyProtectedMember
        if value_render_option not in get_args(VALUE_RENDER_OPT):
            raise ValueError('')

        row, col = self._rowcol(data, head)

        if row is None:
            return self._worksheet.col_values(col, value_render_option=value_render_option)[1:]
        elif isinstance(row, int):
            return self._worksheet.cell(row, col, value_render_option=value_render_option).value
        else:
            return [self._worksheet.cell(it, col, value_render_option=value_render_option).value for it in row]

    def update_cell(self, data: DataIndex, head: str, value: Union[list[str], str]):
        """

        :param data:
        :param head:
        :param value: value to be updated. str type if single field
        :return:
        """
        from gspread.utils import rowcol_to_a1

        row, col = self._rowcol(data, head)
        if row is None:
            if len(value) != len(self.index_value):
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
        self._worksheet.clear()

    def update(self, range_name, values=None, **kwargs):
        self._worksheet.update(range_name, values, **kwargs)

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self._worksheet.get_all_records())

    def to_polars(self) -> pl.DataFrame:
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
                 service_account_path: PathLike):
        """

        :param name: name of the spreadsheet
        :param service_account_path: The path to the service account json file
        """

        self._client = gspread.service_account(filename=service_account_path)

        self._sheet: gspread.Spreadsheet = self._client.open(name)
        self._worksheets: list[gspread.Worksheet] = self._sheet.worksheets()

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

    def has_worksheet(self, title: WorkPageName) -> bool:
        """if has the worksheet, implement also in ``__contains__()``"""
        for w in self._worksheets:
            if w.title == title:
                return True
        return False

    def get_worksheet(self, title: WorkPageName, index_col_name: str = 'Data') -> GoogleWorkSheet:
        """get the worksheet. implement also in ``__get_item__()``"""
        for w in self._worksheets:
            if w.title == title:
                return GoogleWorkSheet(w, index_col_name)
        raise ValueError()


# ========= #
# Utilities #
# ========= #

def upload_dataframe_to_spreadsheet(df: DataFrame, gspread_name: str, worksheet_name: str) -> None:
    """
    Upload a dataframe to a gspread worksheet

    :param df: DataFrame
    :param gspread_name: spreadsheet name
    :param worksheet_name: worksheet name under the spreadsheet
    :return:
    """
    gs = GoogleSpreadSheet(gspread_name)
    spreadsheet = gs._sheet

    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    if worksheet_name not in gs.worksheet_list:
        spreadsheet.add_worksheet(title=worksheet_name, rows=df.shape[0], cols=len(df.columns))
        fprint(f'ADD WORKSHEET: {worksheet_name}')
        gs = GoogleSpreadSheet(gspread_name)  # refresh page

    worksheet = gs[worksheet_name]
    worksheet.clear()

    data = [df.columns] + [field for field in df.iter_rows()]
    worksheet.update(data)
