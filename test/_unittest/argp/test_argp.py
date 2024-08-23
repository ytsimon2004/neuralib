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
        self.assertIsNone(opt.a) # should it be a []

        opt = parse_args(Opt(), ['-a=1'])
        self.assertListEqual(opt.a, ['1'])

        opt = parse_args(Opt(), ['-a=1', '-a=2'])
        self.assertListEqual(opt.a, ['1', '2'])

    def test_list_type_infer(self):
        class Opt:
            a: list[int] = argument(metavar='...', nargs='*', action='extend')

        opt = parse_args(Opt(), ['12', '34'])
        self.assertListEqual(opt.a, [12, 34])


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
