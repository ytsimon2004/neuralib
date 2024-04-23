import unittest


class TestUtilFunc(unittest.TestCase):

    def test_check_attrs_in_clazz(self):
        from neuralib.util.util_clazz import check_attrs_in_clazz

        class Parent:
            a: int
            b: bool
            c: str

        class Child(Parent):
            d: float
            e: dict

        self.assertTrue(check_attrs_in_clazz(Parent, 'a'))
        self.assertFalse(check_attrs_in_clazz(Parent, 'd'))
        self.assertTrue(check_attrs_in_clazz(Child, 'a'))
        self.assertTrue(check_attrs_in_clazz(Child, 'e'))

    def test_key_from_value(self):
        from neuralib.util.utils import key_from_value

        dy = dict(
            a=[1, 2, 3],
            b=5,
            c=(4, 5, 3),
            d=30.5
        )

        self.assertEqual(key_from_value(dy, 3), ['a', 'c'])
        self.assertEqual(key_from_value(dy, 5), ['b', 'c'])
        self.assertEqual(key_from_value(dy, 30.5), 'd')

        with self.assertRaises(KeyError):
            key_from_value(dy, 100)
