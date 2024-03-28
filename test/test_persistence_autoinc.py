import random
from pathlib import Path

from neuralib.persistence import persistence


@persistence.persistence_class
class Example:
    a: int = persistence.field(validator=True, filename=True)
    b: int = persistence.autoinc_field()
    c: int

    def compute(self) -> 'Example':
        self.c = random.randint(0, self.a)
        return self

    def with_b(self, b: int) -> 'Example':
        self.b = b
        return self


handler = persistence.PickleHandler(Example, Path('../../rscvp2/test'))
print(persistence.as_dict(handler.save_persistence(Example(10).compute())))
print(persistence.as_dict(handler.load_persistence(Example(10).with_b(1))))
