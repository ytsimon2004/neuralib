import numpy as np

from neuralib.persistence import persistence
from neuralib.argp import AbstractParser
from rscvp.util.cli import CommonOptions
from rscvp.util.cli_presistence import PersistenceOptions


@persistence.persistence_class
class TestCache:
    used_exp_date: str = persistence.field(validator=True, filename=True)
    used_animal: str = persistence.field(validator=True, filename=True)

    data: np.ndarray


class TestPersistence(AbstractParser, CommonOptions, PersistenceOptions[TestCache]):

    def post_parsing(self):
        self.extend_src_path(self.EXP_DATE, self.ANIMAL, self.REC_TYPE, self.USERNAME)

    def run(self):
        cache = self.load_cache()
        print(cache.data)

    def empty_cache(self) -> TestCache:
        return TestCache(used_exp_date=self.EXP_DATE, used_animal=self.ANIMAL)

    def _compute_cache(self, cache: TestCache) -> TestCache:
        cache.data = np.array([1, 3, 4, 5, 6])
        print('1234😳')
        return cache


if __name__ == '__main__':
    t = TestPersistence()
    t.main()
