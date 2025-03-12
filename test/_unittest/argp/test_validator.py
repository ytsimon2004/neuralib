import unittest

from neuralib.argp import (
    argument, parse_args, int_tuple_type, str_tuple_type, as_dict
)


class TestValidator(unittest.TestCase):

    def test_greater_than(self):
        class Opt:
            a: int = argument('-a', gt=10)

        with self.assertRaises(ValueError) as err:
            parse_args(Opt(), ['-a', '10'])
        print(err.exception)

        parse_args(Opt(), ['-a', '11'])

    def test_less_than(self):
        class Opt:
            a: int = argument('-a', lt=10)

        with self.assertRaises(ValueError) as err:
            parse_args(Opt(), ['-a', '15'])
        print(err.exception)

        parse_args(Opt(), ['-a', '9'])

    def test_in_range(self):
        class Opt:
            a: int = argument('-a', gt=100, lt=150)

        with self.assertRaises(ValueError) as err:
            parse_args(Opt(), ['-a', '15'])

        print(err.exception)

    def test_length_validator(self):
        class Opt:
            a: tuple[int, ...] = argument('-a', type=int_tuple_type, max_length=2)
            b: tuple[str, ...] = argument('-b', type=str_tuple_type, min_length=3)
            c: int = argument('-c', max_length=3)

        with self.assertRaises(ValueError) as err:
            parse_args(Opt(), ['-a', '11,22,33'])
        print(err.exception)

        with self.assertRaises(ValueError) as err:
            parse_args(Opt(), ['-b', '11,22'])
        print(err.exception)

        with self.assertWarns(UserWarning) as warn:
            parse_args(Opt(), ['-c', '123'])
        print(str(warn.warning))

    def test_custom_validator(self):

        def validate_elements(value):
            if value[0] == 'red':
                print('GOT first red')

            if value[1] != 'green':
                print('FAIL second green')

        class Opt:
            a: tuple[int, ...] = argument('-a', type=int_tuple_type, validator=lambda it: it[0] > 100, default=(150, 10))
            b: tuple[str, ...] = argument('-b', type=str_tuple_type, validator=validate_elements)

        with self.assertRaises(ValueError) as err:
            parse_args(Opt(), ['-a', '90,10'])
        print(err.exception)

        opt = parse_args(Opt(), ['-a', '101,200', '-b', 'red,orange'])
        print(as_dict(opt))
