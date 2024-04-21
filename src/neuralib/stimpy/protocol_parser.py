from __future__ import annotations

import io
import re
from typing import Any

import polars as pl

from neuralib.stimpy.math_eval import evaluate_string

__all__ = [
    'EmptyProtocolError',
    'UnassignedVariableError',
    'UndefinedVariableError',
    'VariableAlreadyDeclaredError',
    #
    'remove_comments_and_strip',
    'lines_to_variables_dict',
    'eval_assignments',
    'eval_dataframe',
    'process_dataframe',
    'generate_extended_dataframe',
]


class ProtocolError(ValueError):
    """Custom Error class for protocol errors"""

    def __init__(self, message: str, line: int | None = None):
        """
        :param message: error message
        :param line: line number
        """
        if line is not None:
            message += f' on protocol line {line}'
        super().__init__(message)
        self.line = line


class VariableAlreadyDeclaredError(ProtocolError):
    def __init__(self, var_name: str, line: int = None):
        super().__init__(f"Variable {var_name} already declared", line=line)


class EmptyProtocolError(ProtocolError):
    def __init__(self):
        super().__init__("empty protocol")


class UndefinedVariableError(ProtocolError):
    def __init__(self, var_name: str, line: int = None):
        super().__init__(f"undefined variable {var_name}", line=line)


class UnassignedVariableError(ProtocolError):
    def __init__(self, var_name: str, line: int = None):
        super().__init__(f"unassigned variable {var_name}", line=line)


# ================= #

def remove_comments_and_strip(lines: list[str]) -> list[str]:
    """Extracts assignments for all lines given,
    removes the eventual comment and strip leading/trailing spacings,
    discards lines without assignment"""
    stripped_lines = []
    for line in lines:
        stripped_line, _, _ = line.partition('#')
        stripped_line = stripped_line.strip()
        if stripped_line != '':
            stripped_lines.append(stripped_line)

    return stripped_lines


def lines_to_variables_dict(lines: list[str]) -> dict[str, str]:
    """Identify assignments (=)
    and create dict {variable_name : string_to_eval}"""
    variables_dict = {}
    variables_loc = {}

    current_variable = ''
    current_string = ''

    assign_pattern = re.compile(r'^([A-Za-z0-9_]+)\s*=\s*(.*)\s*$')

    for line, content in enumerate(lines):

        if (match := assign_pattern.match(content)) is not None:
            current_variable = match.group(1).strip()
            current_string = match.group(2).strip()

            if current_variable in variables_dict:
                raise VariableAlreadyDeclaredError(current_variable, line=line + 1)

            variables_dict[current_variable] = current_string
            variables_loc[current_variable] = line + 1
        else:
            if current_string != '':
                current_string += "\n"

            current_string += content.strip()
            variables_dict[current_variable] = current_string

    check_variables_dict_for_errors(variables_dict, variables_loc)
    return variables_dict


def check_variables_dict_for_errors(variables_dict: dict[str, str], variables_loc: dict[str, int]):
    if len(variables_dict) == 0:
        raise EmptyProtocolError()

    for key, value in variables_dict.items():
        if key == '':
            raise UndefinedVariableError(key, line=variables_loc.get(key, None))
        elif value == '':
            raise UnassignedVariableError(key, line=variables_loc.get(key, None))


def eval_assignments(variables_dict: dict[str, str], *,
                     dataframe_var=('visual_stimuli',)) -> dict[str, Any]:
    """Evaluates the dictionary's values and returns the evaluated dict"""
    evaluated_dict = {}
    for key in variables_dict:
        str_val = variables_dict[key]

        if key in dataframe_var:
            typed_val = process_dataframe(eval_dataframe(str_val))
        else:
            try:
                typed_val = eval(str_val)
            except (ValueError, SyntaxError, NameError):
                typed_val = str_val

        evaluated_dict[key] = typed_val

    return evaluated_dict


# ==================== #
# DataFrame Processing #
# ==================== #

def eval_dataframe(dataframe_string: str) -> pl.DataFrame:
    """Evaluates the dataframe contained in the string
    Returns True and a polar dataframe object if the string contains
    a dataframe, else returns False and None
    """
    dataframe_string = re.sub(r' +', ';', dataframe_string)
    return pl.read_csv(io.StringIO(dataframe_string), has_header=True, separator=';', use_pyarrow=True)


def process_dataframe(dataframe: pl.DataFrame) -> pl.DataFrame:
    """1. Eventually generates extra lines from {i} one-liner magic in 'n' column.
       2. Orders the 'n' column.
    """
    ret = generate_extended_dataframe(dataframe)
    return ret


def generate_extended_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Undoes the magic n-i one-liner commands and generates needed extra lines. i=0 on line of declaration.
    Example:
            n        dur     xc  yc      c   width   height       func     sf      ori
            0           1     0   1.0   10     10  10+10*sin({t})    sin    0.01    30
            1-2         2     0   0     1.0    50       50         sin    0.01    60*{i}
        becomes
            n        dur     xc  yc      c   width   height       func     sf      ori
            0           1     0   1.0   10     10  10+10*sin({t})    sin    0.01    30
            1           2     0   0     1.0    50       50         sin    0.01    60*0
            2           2     0   0     1.0    50       50         sin    0.01    60*1
    """

    if 'n' not in df.columns or df.schema['n'].is_integer():
        return df

    n = pl.col('n')
    ef = df.filter(n.str.contains('-'))
    vf = df.filter(~n.str.contains('-')).with_columns(n.cast(pl.Int64))

    def _extend_row(row: dict[str, str]) -> dict[str, list[str]]:
        a, _, b = row['n'].partition('-')
        a = int(a)
        b = int(b)
        return {
            k: list(range(a, b + 1)) if k == 'n' else (
                _extend_expr(k, v, a, b) if isinstance(v, str) else [v] * (b - a + 1)
            )
            for k, v in row.items()
        }

    do_not_eval = set()
    eval_as_int = set()

    def _extend_expr(k: str, expr: str, a: int, b: int) -> list[str]:
        try:
            if k not in do_not_eval:
                ret = [evaluate_string(expr.replace('{i}', str(i))) for i in range(a, b + 1)]
                eval_as_int.add(k)
                return ret
        except:
            do_not_eval.add(k)

        return [expr.replace('{i}', str(i)) for i in range(a, b + 1)]

    ef = [
        _extend_row(it)
        for it in ef.iter_rows(named=True)
    ]

    vf = vf.with_columns([
        pl.col(col).cast(pl.Int64)
        for col in eval_as_int
    ])

    return pl.concat([vf, pl.concat(list(map(pl.DataFrame, ef)))]).sort(n)
