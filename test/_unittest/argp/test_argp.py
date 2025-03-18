import unittest
from pathlib import Path
from typing import Optional, Literal

from neuralib.argp import *


class TypeAnnotationTest(unittest.TestCase):
    def test_bool(self):
        class Opt:
            a: bool = argument('-a')

        opt = parse_args(Opt(), ['-a'])
        self.assertTrue(opt.a)

        opt = parse_args(Opt(), [])
        self.assertFalse(opt.a)

    def test_bool_set_false(self):
        class Opt:
            a: bool = argument('-a', default=True)

        opt = parse_args(Opt(), ['-a'])
        self.assertFalse(opt.a)

        opt = parse_args(Opt(), [])
        self.assertTrue(opt.a)

    def test_str(self):
        class Opt:
            a: str = argument('-a')

        opt = parse_args(Opt(), ['-a', 'test'])
        self.assertEqual(opt.a, 'test')
        opt = parse_args(Opt(), [])
        self.assertIsNone(opt.a)

    def test_optional_str(self):
        class Opt:
            a: Optional[str] = argument('-a')

        opt = parse_args(Opt(), ['-a', 'test'])
        self.assertEqual(opt.a, 'test')
        opt = parse_args(Opt(), [])
        self.assertIsNone(opt.a)

    def test_optional_pipeline_int(self):
        class Opt:
            a: int | None = argument('-a')

        opt = parse_args(Opt(), ['-a', '1'])
        self.assertEqual(opt.a, 1)
        opt = parse_args(Opt(), [])
        self.assertIsNone(opt.a)

    def test_int(self):
        class Opt:
            a: int = argument('-a')

        opt = parse_args(Opt(), ['-a', '10'])
        self.assertEqual(opt.a, 10)
        opt = parse_args(Opt(), [])
        self.assertIsNone(opt.a)

    def test_float(self):
        class Opt:
            a: float = argument('-a')

        opt = parse_args(Opt(), ['-a', '10.321'])
        self.assertEqual(opt.a, 10.321)
        opt = parse_args(Opt(), [])
        self.assertIsNone(opt.a)

    def test_path(self):
        class Opt:
            a: Path = argument('-a')

        opt = parse_args(Opt(), ['-a', 'test_argp.py'])
        self.assertEqual(opt.a, Path('test_argp.py'))
        opt = parse_args(Opt(), [])
        self.assertIsNone(opt.a)

    def test_literal(self):
        class Opt:
            a: Literal['A', 'B'] = argument('-a')

        opt = parse_args(Opt(), ['-a', 'A'])
        self.assertEqual(opt.a, 'A')

        opt = parse_args(Opt(), [])
        self.assertIsNone(opt.a)

        with self.assertRaises(SystemExit):
            parse_args(Opt(), ['-a', 'C'])

    def test_optional_literal(self):
        class Opt:
            a: Optional[Literal['A', 'B']] = argument('-a')

        opt = parse_args(Opt(), ['-a', 'A'])
        self.assertEqual(opt.a, 'A')

        opt = parse_args(Opt(), [])
        self.assertIsNone(opt.a)

        with self.assertRaises(SystemExit):
            parse_args(Opt(), ['-a', 'C'])

    def test_literal_with_choice(self):
        class Opt:
            a: Literal['A', 'B'] = argument('-a', choices=('A', 'B', 'C'))

        opt = parse_args(Opt(), ['-a', 'C'])
        self.assertEqual(opt.a, 'C')

    def test_list_type_extend(self):
        class Opt:
            a: list[str] = argument(metavar='...', nargs='*', action='extend')

        opt = parse_args(Opt(), [])
        self.assertListEqual(opt.a, [])

        opt = parse_args(Opt(), ['12', '34'])
        self.assertListEqual(opt.a, ['12', '34'])

    def test_list_type_append(self):
        class Opt:
            a: list[str] = argument('-a', action='append')

        opt = parse_args(Opt(), [])
        # self.assertListEqual(opt.a, [])
        self.assertIsNone(opt.a)  # should it be a []

        opt = parse_args(Opt(), ['-a=1'])
        self.assertListEqual(opt.a, ['1'])

        opt = parse_args(Opt(), ['-a=1', '-a=2'])
        self.assertListEqual(opt.a, ['1', '2'])

    def test_list_type_infer(self):
        class Opt:
            a: list[int] = argument(metavar='...', nargs='*', action='extend')

        opt = parse_args(Opt(), ['12', '34'])
        self.assertListEqual(opt.a, [12, 34])


class TestValidator(unittest.TestCase):
    def test_validator(self):
        class Opt:
            a: str = argument('-a', validator=lambda it: len(it) > 0)

        opt = parse_args(Opt(), ['-a=1'])
        self.assertEqual(opt.a, '1')

        with self.assertRaises(SystemExit):
            parse_args(Opt(), ['-a='])

        opt.a = '2'
        self.assertEqual(opt.a, '2')

        with self.assertRaises(ValueError):
            opt.a = ''

        with self.assertRaises(ValueError):
            opt.a = None

    def test_validator_tuple(self):
        class Opt:
            a: tuple[str, str] = argument('-a', type=str_tuple_type, validator=lambda it: len(it) == 2)
            b: tuple[int, ...] | None = argument('-b', type=int_tuple_type,
                                                 validator=lambda it: it is None or all([i < 5 for i in it]))

        with self.assertRaises(SystemExit):
            parse_args(Opt(), ['-a=10,2,3'])

        with self.assertRaises(SystemExit):
            parse_args(Opt(), ['-b=6,2'])

    def test_validate_on_parse(self):
        class Opt:
            a: str = argument('-a', validator=lambda it: len(it) > 0, validate_on_set=False)

        opt = parse_args(Opt(), ['-a=1'])
        self.assertEqual(opt.a, '1')

        with self.assertRaises(SystemExit):
            parse_args(Opt(), ['-a='])

        opt.a = '2'
        self.assertEqual(opt.a, '2')
        opt.a = ''
        self.assertEqual(opt.a, '')
        opt.a = None
        self.assertIsNone(opt.a, None)

    def test_validator_with_type_caster(self):
        class Opt:
            a: int = argument('-a', validator=lambda it: it >= 0)

        opt = parse_args(Opt(), ['-a=1'])
        self.assertEqual(opt.a, 1)

        with self.assertRaises(SystemExit):
            parse_args(Opt(), ['-a=-1'])

        opt.a = 1
        self.assertEqual(opt.a, 1)

        with self.assertRaises(ValueError):
            opt.a = -1

    def test_validate_on_set_on_normal_attr(self):
        class Opt:
            a: str = argument('-a', validator=lambda it: len(it) > 0)

        with self.assertRaises(SystemExit):
            parse_args(Opt(), ['-a='])

        with self.assertRaises(ValueError):
            Opt().a = ''

    def test_validate_on_set_on_protected_attr(self):
        class Opt:
            _a: str = argument('-a', validator=lambda it: len(it) > 0)

        with self.assertRaises(SystemExit):
            parse_args(Opt(), ['-a='])

        opt = Opt()
        opt._a = ''
        self.assertEqual(opt._a, '')


class TestValidateBuilder(unittest.TestCase):
    def test_str_in_range(self):
        class Opt:
            a: str = argument('-a', validator.str.length_in_range(2, None))
            b: str = argument('-b', validator.str.length_in_range(None, 2))
            c: str = argument('-c', validator.str.length_in_range(2, 4))

        opt = Opt()
        opt.a = '12'
        opt.b = '12'
        opt.c = '12'

        with self.assertRaises(ValueError):
            opt.a = ''
        with self.assertRaises(ValueError):
            opt.b = '1234'
        with self.assertRaises(ValueError):
            opt.c = ''
        with self.assertRaises(ValueError):
            opt.c = '12345678'

    def test_int_in_range(self):
        class Opt:
            a: int = argument('-a', validator.int.in_range(2, None))
            b: int = argument('-b', validator.int.in_range(None, 2))
            c: int = argument('-c', validator.int.in_range(2, 4))

        opt = Opt()
        opt.a = 2
        opt.b = 2
        opt.c = 2

        with self.assertRaises(ValueError):
            opt.a = 0
        with self.assertRaises(ValueError):
            opt.b = 10
        with self.assertRaises(ValueError):
            opt.c = 0
        with self.assertRaises(ValueError):
            opt.c = 10

    def test_float_positive(self):
        class Opt:
            a: float = argument('-a', validator.float.positive())
            b: float = argument('-b', validator.float.negative())

        opt = Opt()

        opt.a = 10
        with self.assertRaises(ValueError):
            opt.a = -10

        opt.b = -10
        with self.assertRaises(ValueError):
            opt.b = 10

        opt.a = 0
        opt.b = 0

    def test_float_allow_nan(self):
        class Opt:
            a: float = argument('-a', validator.float.allow_nan(True))
            b: float = argument('-b', validator.float.allow_nan(False))

        opt = Opt()
        opt.a = float('nan')

        with self.assertRaises(ValueError):
            opt.b = float('nan')

    def test_float_allow_nan_then(self):
        class Opt:
            a: float = argument('-a', validator.float.allow_nan(False).positive())

        opt = Opt()
        opt.a = 10

        with self.assertRaises(ValueError):
            opt.a = float('nan')

        with self.assertRaises(ValueError):
            opt.a = -10

    def test_list_element_type(self):
        class Opt:
            a: list[int] = argument('-a', validator.list(int))

        opt = Opt()
        opt.a = []
        opt.a = [1, 2]

        with self.assertRaises(ValueError):
            opt.a = ['a']

    def test_list_element_validating(self):
        class Opt:
            a: list[int] = argument('-a', validator.list(int).on_item(validator.int.positive(include_zero=True)))

        opt = Opt()
        opt.a = []
        opt.a = [1, 2]

        with self.assertRaises(ValueError):
            opt.a = [-1]

        with self.assertRaises(ValueError) as capture:
            opt.a = [1, -1]

        self.assertEqual(capture.exception.args[0],
                         'at index 1, not a non-negative value : -1')

    def test_tuple_element_type(self):
        class Opt:
            a: tuple[str, int, float] = argument(
                '-a',
                validator.tuple(str, int, float)
            )

        opt = Opt()
        opt.a = ('', 0, 0.0)

        with self.assertRaises(ValueError):
            opt.a = ()

        with self.assertRaises(ValueError):
            opt.a = ('', 0)

        with self.assertRaises(ValueError):
            opt.a = ('', 0, 0)

        with self.assertRaises(ValueError):
            opt.a = (0, 0)

    def test_tuple_element_type_var_length(self):
        class Opt:
            a: tuple[str, int, ...] = argument(
                '-a',
                validator.tuple(str, int, ...)
            )

        opt = Opt()
        opt.a = ('', 0)
        opt.a = ('', 0, 0)
        opt.a = ('', 0, 0, 0)

        with self.assertRaises(ValueError):
            opt.a = ()

        with self.assertRaises(ValueError):
            opt.a = ('',)

        with self.assertRaises(ValueError):
            opt.a = (0, 0)

    def test_tuple_element_validating(self):
        class Opt:
            a: tuple[str, int, float] = argument(
                '-a',
                validator.tuple(str, int, float) \
                    .on_item(0, validator.str.length_in_range(None, 2)) \
                    .on_item(1, validator.int.in_range(0, 10))
            )

        opt = Opt()
        opt.a = ('', 0, 0.0)

        with self.assertRaises(ValueError) as capture:
            opt.a = ('1234', 0, 0.0)
        self.assertEqual(capture.exception.args[0],
                         'at index 0, str length over 2: "1234"')

        with self.assertRaises(ValueError) as capture:
            opt.a = ('12', 100, 0.0)
        self.assertEqual(capture.exception.args[0],
                         'at index 1, value out of range [0, 10]: 100')

    def test_optional(self):
        class Opt:
            a: int = argument('-a', validator.int)
            b: int | None = argument('-b', validator.int.optional())
            c: int | None = argument('-c', validator.any(validator.optional(), validator.int))

        opt = Opt()
        opt.a = 0
        opt.b = 0
        opt.c = 0

        with self.assertRaises(ValueError):
            opt.a = None
        opt.b = None
        opt.c = None

    def test_any(self):
        class Opt:
            a: int | str = argument('-a', validator.any(
                validator.int.in_range(0, 10),
                validator.str.length_in_range(0, 10)
            ))

        opt = Opt()
        opt.a = 3
        opt.a = '123'

        with self.assertRaises(ValueError) as capture:
            opt.a = 30

        self.assertEqual(capture.exception.args[0],
                         'value out of range [0, 10]: 30')

        with self.assertRaises(ValueError) as capture:
            opt.a = '1' * 13

        self.assertEqual(capture.exception.args[0],
                         'str length out of range [0, 10]: "1111111111111"')

    def test_any_literal(self):
        class Opt:
            a: int | str = argument('-a', (
                    validator.int.in_range(0, 10) | validator.str.length_in_range(0, 10)
            ))

        opt = Opt()
        opt.a = 3
        opt.a = '123'

        with self.assertRaises(ValueError) as capture:
            opt.a = 30

        self.assertEqual(capture.exception.args[0],
                         'value out of range [0, 10]: 30')

        with self.assertRaises(ValueError) as capture:
            opt.a = '1' * 13

        self.assertEqual(capture.exception.args[0],
                         'str length out of range [0, 10]: "1111111111111"')

    def test_all(self):
        class Opt:
            a: int = argument('-a', validator.all(
                validator.int.positive(include_zero=True),
                validator.int.negative(include_zero=True),
            ))

        opt = Opt()
        opt.a = 0

        with self.assertRaises(ValueError) as capture:
            opt.a = 1

        self.assertEqual(capture.exception.args[0],
                         'not a non-positive value : 1')

        with self.assertRaises(ValueError) as capture:
            opt.a = -1

        self.assertEqual(capture.exception.args[0],
                         'not a non-negative value : -1')

    def test_all_literal(self):
        class Opt:
            a: int = argument('-a', (
                    validator.int.positive(include_zero=True) & validator.int.negative(include_zero=True)
            ))

        opt = Opt()
        opt.a = 0

        with self.assertRaises(ValueError) as capture:
            opt.a = 1

        self.assertEqual(capture.exception.args[0],
                         'not a non-positive value : 1')

        with self.assertRaises(ValueError) as capture:
            opt.a = -1

        self.assertEqual(capture.exception.args[0],
                         'not a non-negative value : -1')

    def test_tuple_union_length(self):
        class Opt:
            a: tuple[int, int] | tuple[int, int, int] = argument(
                '-a',
                validator.tuple(int, int) | validator.tuple(int, int, int)
            )

        opt = Opt()
        opt.a = (0, 1)
        opt.a = (0, 1, 2)

        with self.assertRaises(ValueError) as capture:
            opt.a = (0,)
        # print(capture.exception.args)
        # length not match to 2 : (0,); length not match to 3 : (0,)

        with self.assertRaises(ValueError) as capture:
            opt.a = (0, 1, 2, 3)
        # print(capture.exception.args)
        # length not match to 2 : (0, 1, 2, 3); length not match to 3 : (0, 1, 2, 3)

    def test_nested_list(self):
        class Opt:
            a: list[tuple[int, list[list[int]]]] = argument('-a', validator.list(
                validator.tuple(int, None)
                .on_item(0, validator.int)
                .on_item(1, validator.list().on_item(validator.list(int)))
            ))

        opt = Opt()
        opt.a = []
        opt.a = [(0, [[0]]), (1, [[1]])]

        with self.assertRaises(ValueError) as capture:
            opt.a = [([[0]])]
        self.assertEqual(capture.exception.args[0],
                         'not a tuple : [[0]]')
        with self.assertRaises(ValueError) as capture:
            opt.a = [(0, [0])]
        self.assertEqual(capture.exception.args[0],
                         'at index 1, at index 0, not a list : 0')

    def test_tuple_on_multiple_item(self):
        class Opt:
            a: tuple[int, float, int, float] = argument(
                '-a',
                validator.tuple() \
                    .on_item([0, 2], validator.int.positive()) \
                    .on_item(1, v := validator.float.positive()) \
                    .on_item(3, v)
            )

        opt = Opt()
        opt.a = (1, 1, 1, 1)

        with self.assertRaises(ValueError) as capture:
            opt.a = (1, 1, -1, 1)
        with self.assertRaises(ValueError) as capture:
            opt.a = (1, 1, 1, -1)

    def test_tuple_fix_length(self):
        class Opt:
            a: tuple[int, int] = argument(
                '-a',
                validator.tuple(2)
            )

        opt = Opt()
        opt.a = (0, 1)

        with self.assertRaises(ValueError) as capture:
            opt.a = (0,)

        with self.assertRaises(ValueError) as capture:
            opt.a = (0, 1, 2)

        opt.a = ('0', '1')

    def test_tuple_on_all_item(self):
        class Opt:
            a: tuple[int, int] = argument(
                '-a',
                validator.tuple(2).on_item(None, validator.int)
            )

        opt = Opt()
        opt.a = (0, 1)

        with self.assertRaises(ValueError) as capture:
            opt.a = ('0', '1')


class WithDefaultTest(unittest.TestCase):
    def test_bool(self):
        class Opt:
            a: bool = argument('-a')

        opt = with_defaults(Opt())
        self.assertFalse(opt.a)

    def test_bool_set_false(self):
        class Opt:
            a: bool = argument('-a', default=True)

        opt = with_defaults(Opt())
        self.assertTrue(opt.a)

    def test_str(self):
        class Opt:
            a: str = argument('-a')

        opt = with_defaults(Opt())
        self.assertIsNone(opt.a)

    def test_default_str(self):
        class Opt:
            a: str = argument('-a', default='default')

        opt = with_defaults(Opt())
        self.assertEqual(opt.a, 'default')

    def test_int(self):
        class Opt:
            a: int = argument('-a')

        opt = with_defaults(Opt())
        self.assertIsNone(opt.a)

    def test_default_int(self):
        class Opt:
            a: int = argument('-a', default=101)

        opt = with_defaults(Opt())
        self.assertEqual(opt.a, 101)

    def test_float(self):
        class Opt:
            a: float = argument('-a')

        opt = with_defaults(Opt())
        self.assertIsNone(opt.a)

    def test_default_float(self):
        class Opt:
            a: float = argument('-a', default=3.14)

        opt = with_defaults(Opt())
        self.assertEqual(opt.a, 3.14)

    def test_literal(self):
        class Opt:
            a: Literal['A', 'B'] = argument('-a')

        opt = with_defaults(Opt())
        self.assertIsNone(opt.a)

    def test_default_literal(self):
        class Opt:
            a: Literal['A', 'B'] = argument('-a', default='C')

        opt = with_defaults(Opt())
        self.assertEqual(opt.a, 'C')


class AsDictTest(unittest.TestCase):
    def test_emtpy(self):
        class Opt:
            a: str = argument('-a', default='default')

        self.assertDictEqual(as_dict(Opt()), {})

    def test_as_dict(self):
        class Opt:
            a: str = argument('-a', default='default')

        opt = with_defaults(Opt())
        self.assertDictEqual(as_dict(opt), {'a': 'default'})


class AbstractParserTest(unittest.TestCase):
    def test_exit_on_error(self):
        class Main(AbstractParser):
            a: str = argument('-a', default='default')

        with self.assertRaises(SystemExit):
            Main().main(['-b'], exit_on_error=True)

        with self.assertRaises(RuntimeError):
            Main().main(['-b'], exit_on_error=False)


class CommandParserTest(unittest.TestCase):
    def test_command_parser(self):
        class P1(AbstractParser):
            a: str = argument('-a', default='default')

        class P2(AbstractParser):
            a: str = argument('-a', default='default')

        parsers = dict(a=P1, b=P2)
        opt = parse_command_args(parsers, ['a'], run_main=False)
        self.assertIsInstance(opt, P1)
        opt = parse_command_args(parsers, ['b'], run_main=False)
        self.assertIsInstance(opt, P2)
        opt = parse_command_args(parsers, [], run_main=False)
        self.assertIsNone(opt)

    def test_command_parser_main(self):
        class P1(AbstractParser):
            a: str = argument('-a', default='default')

        class P2(AbstractParser):
            a: str = argument('-a', default='default')

        parsers = dict(a=P1, b=P2)
        opt = parse_command_args(parsers, ['a'])
        self.assertIsInstance(opt, P1)
        self.assertEqual('default', opt.a)
        opt = parse_command_args(parsers, ['b'])
        self.assertIsInstance(opt, P2)
        self.assertEqual('default', opt.a)
        opt = parse_command_args(parsers, [])
        self.assertIsNone(opt)


if __name__ == '__main__':
    unittest.main()
