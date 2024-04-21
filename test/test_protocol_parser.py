import unittest

import polars as pl
from polars.testing import assert_frame_equal

from neuralib.stimpy.protocol_parser import (
    remove_comments_and_strip,
    lines_to_variables_dict,
    EmptyProtocolError,
    UnassignedVariableError,
    UndefinedVariableError,
    VariableAlreadyDeclaredError,
    eval_assignments, eval_dataframe, generate_extended_dataframe
)


class TestProtocolParser(unittest.TestCase):
    """Test class for parser"""

    def test_remove_comments_and_strip(self):
        lines = [
            " #4",
            "#2",
            "var_1 = 3 # x",
            "# var_2 = 4 # var_3 = 5 # x",
            "var_2 =",
            "  a b c",
            "  d e f",
        ]
        stripped_prot = remove_comments_and_strip(lines)
        expected_result = [
            "var_1 = 3",
            "var_2 =",
            "a b c",
            "d e f"
        ]
        self.assertListEqual(stripped_prot, expected_result)

    def test_lines_to_variables_dict_success(self):
        assignments = [
            "var_1 = 3",
            "var_2 =",
            "a b c",
            "d>=1 e<=2 f"
        ]  # "d e f"
        variables_dict = lines_to_variables_dict(assignments)
        expected_result = {
            "var_1": "3",
            "var_2": "a b c\nd>=1 e<=2 f"
        }
        self.assertDictEqual(variables_dict, expected_result)

    def test_lines_to_variables_dict_empty_raises_error(self):
        with self.assertRaises(EmptyProtocolError):
            lines_to_variables_dict([])

    def test_lines_to_variables_dict_unassigned_raises_error(self):
        with self.assertRaises(UnassignedVariableError):
            lines_to_variables_dict(["a="])

    def test_lines_to_variables_dict_undefined_variable_raises_error(self):
        with self.assertRaises(UndefinedVariableError):
            lines_to_variables_dict(["=a"])

    def test_lines_to_variables_dict_multiple_declaration_raises_error(self):
        with self.assertRaises(VariableAlreadyDeclaredError):
            lines_to_variables_dict(["a=1", "a=2"])

    def test_eval_assignments(self):
        dummy_assignment_dict = {
            "var1": "1",
            "var2": "2.0",
            "var3": "True",
            "var4": "astring",
            "var5": "[1,2]",
            "var6": "{'v1': 1}",
            "var7": "dur ori other\n1 2 3",
        }
        evaluated_dict = eval_assignments(dummy_assignment_dict, dataframe_var=['var7'])
        self.assertEqual(evaluated_dict["var1"], int(1))
        self.assertEqual(evaluated_dict["var2"], float(2))
        self.assertEqual(evaluated_dict["var3"], True)
        self.assertEqual(evaluated_dict["var4"], "astring")
        self.assertEqual(evaluated_dict["var5"], [1, 2])
        self.assertEqual(evaluated_dict["var6"], {"v1": 1})
        df = pl.DataFrame(data=dict(dur=[1], ori=[2], other=[3]))
        assert_frame_equal(evaluated_dict["var7"], df, check_dtype=False)

    def test_parse_dataframe(self):
        data_frame_line = """\
n   img   g   dur len   xc  yc   c   width   height   mask
0    0    0   2   3    -15  0   1.0   15      15    circle
1    1    0   2   3     15  0   0.5   15      15    circle
2    0    1   2   3     15  0   1.0   15      15    circle
3-4    1    1   2   3    -15  0   0.5   15      15    circle
"""
        df = eval_dataframe(data_frame_line)
        assert_frame_equal(df, pl.DataFrame(dict(
            n=['0', '1', '2', '3-4'],
            img=[0, 1, 0, 1],
            g=[0, 0, 1, 1],
            dur=[2, 2, 2, 2],
            len=[3, 3, 3, 3],
            xc=[-15, 15, 15, -15],
            yc=[0, 0, 0, 0],
            c=[1.0, 0.5, 1.0, 0.5],
            width=[15, 15, 15, 15],
            height=[15, 15, 15, 15],
            mask=['circle', 'circle', 'circle', 'circle'],
        )))

    def test_generate_extended_dataframe(self):
        df = pl.DataFrame(dict(
            n=['0', '2-4', '1', ],
            v=['1', '{i}+3', '2', ],
            u=[0, 1, 2]
        ))
        df = generate_extended_dataframe(df)

        result = pl.DataFrame({
            'n': [0, 1, 2, 3, 4],
            'v': [1, 2, 5, 6, 7],
            'u': [0, 2, 1, 1, 1]
        })

        print(result)
        assert_frame_equal(df, result)
