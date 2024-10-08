import unittest

import numpy as np

from neuralib.util.deprecation import deprecated_class, deprecated_func, deprecated_aliases
from neuralib.util.verbose import publish_annotation


class TestUtilFunc(unittest.TestCase):

    def test_cls_hasattr(self):
        from neuralib.util.utils import cls_hasattr

        class Parent:
            a: int
            b: bool
            c: str

        class Child(Parent):
            d: float
            e: dict

        self.assertTrue(cls_hasattr(Parent, 'a'))
        self.assertFalse(cls_hasattr(Parent, 'd'))
        self.assertTrue(cls_hasattr(Child, 'a'))
        self.assertTrue(cls_hasattr(Child, 'e'))

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

    def test_deprecate_class(self):
        @deprecated_class(new='B', remarks='TEST REMARKS', removal_version='v0.0.10')
        class A:
            ...

        with self.assertWarns(DeprecationWarning) as warns:
            A()

        self.assertIn(
            'TestUtilFunc.test_deprecate_class.<locals>.A is deprecated and will be removed in a future release(after version v0.0.10). Please use "B" instead. NOTE: TEST REMARKS.',
            str(warns.warning))

        self.assertIn('TEST REMARKS', str(warns.warning))

    def test_deprecate_function(self):
        @deprecated_func(new='new()', removal_version='v1.0.0')
        def test_deprecate():
            ...

        with self.assertWarns(DeprecationWarning) as warns:
            test_deprecate()

        self.assertIn(
            'TestUtilFunc.test_deprecate_function.<locals>.test_deprecate is deprecated and will be removed in a future release(after version v1.0.0). Please use "new()" instead.',
            str(warns.warning)
        )

    def test_deprecate_aliases(self):
        @deprecated_aliases(old='new')
        def test(new: np.ndarray):
            return np.max(new)

        with self.assertWarns(DeprecationWarning) as warns:
            ret = test(old=np.array([1, 2, 3, 4, 5]))

        self.assertIn('"old" is deprecated and will be removed in future version. Use "new" instead',
                      str(warns.warning))
        self.assertEqual(ret, 5)

    def test_deprecate_aliases_runtime_err(self):
        @deprecated_aliases(old='new_arg')
        def test(new: np.ndarray):
            return np.max(new)

        with self.assertRaises(RuntimeError) as err:
            test()

    def test_deprecate_aliases_value_err(self):
        @deprecated_aliases(old='new')
        def test(new: np.ndarray):
            return np.max(new)

        with self.assertRaises(ValueError) as err:
            test(new=np.array([1, 2, 3]), old=np.array([1, 2, 3]))

    def test_publish_annotation_instance(self):
        @publish_annotation('main', as_attributes=True)
        class Test:
            pass

        @publish_annotation('sup', figure='fig.S1', as_attributes=True)
        def test():
            pass

        self.assertEqual(Test().__publish_level__, 'main')
        self.assertEqual(test.__publish_figure__, 'fig.S1')


if __name__ == '__main__':
    unittest.main()
