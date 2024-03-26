import dataclasses
import unittest

from stimpy._controller.parsing import parse_preferences

from rscvp.util.preference import ZStimpyPreferences


class TestStimpyPreference(unittest.TestCase):
    mydict: dict
    apdict: dict

    @classmethod
    def setUpClass(cls) -> None:
        ap = parse_preferences('simon')
        cls.apdict = {key: ap[key] for key in sorted(ap)}
        cls.mydict = dataclasses.asdict(ZStimpyPreferences.load('stimpy', 'simon'))

        print(cls.apdict)
        print(cls.mydict)

    def test_prefs_equal(self):
        self.assertDictEqual(self.apdict, self.mydict)


if __name__ == '__main__':
    unittest.main()
