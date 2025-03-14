from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, TypeVar, Generic, final

__all__ = ['validator']

T = TypeVar('T')


class ValidatorFailError(ValueError):
    pass


class Validator:
    def __call__(self, value: Any) -> bool:
        return True


class LambdaValidator(Validator):
    def __init__(self, validator: Callable[[Any], bool]):
        self.__validator = validator

    def __call__(self, value: Any) -> bool:
        return self.__validator(value)


@final
class ValidatorBuilder(Validator):
    @property
    def str(self) -> StrValidatorBuilder:
        return StrValidatorBuilder()

    @property
    def int(self) -> IntValidatorBuilder:
        return IntValidatorBuilder()

    @property
    def float(self) -> FloatValidatorBuilder:
        return FloatValidatorBuilder()

    # noinspection PyMethodMayBeStatic
    def tuple(self, *element_type: type[T]) -> TupleValidatorBuilder:
        return TupleValidatorBuilder(element_type)

    # noinspection PyMethodMayBeStatic
    def list(self, element_type: type[T] = None) -> ListValidatorBuilder:
        return ListValidatorBuilder(element_type)

    @classmethod
    def and_(cls, *validator: Callable[[T], bool]) -> AndValidatorBuilder:
        return AndValidatorBuilder(*validator)

    @classmethod
    def or_(cls, *validator: Callable[[T], bool]) -> OrValidatorBuilder:
        return OrValidatorBuilder(*validator)

    @classmethod
    def optional(cls) -> Validator:
        return LambdaValidator(lambda it: it is None)


validator = ValidatorBuilder()


class AbstractTypeValidatorBuilder(Validator, Generic[T]):
    def __init__(self, value_type: type[T] | tuple[type[T], ...] = None):
        self.__value_type = value_type
        self.__validators: list[tuple[Callable[[T], str], Callable[[T], bool]]] = []

    def __call__(self, value: Any) -> bool:
        if self.__value_type is not None and not isinstance(value, self.__value_type):
            raise ValidatorFailError(f'not type of {self.__value_type.__name__} : {value}')

        for message, validator in self.__validators:
            try:
                fail = not validator(value)
            except ValidatorFailError:
                raise
            except BaseException as e:
                raise ValidatorFailError(message(value)) from e
            else:
                if fail:
                    raise ValidatorFailError(message(value))

        return True

    def check(self, validator: Callable[[T], bool], message: str | Callable[[T], str] = None):
        if message is None:
            message = 'validate fail'

        if isinstance(message, str):
            message = message.__mod__

        self.__validators.append((message, validator))

    def optional(self) -> OrValidatorBuilder:
        return OrValidatorBuilder(LambdaValidator(lambda it: it is None), self)


class StrValidatorBuilder(AbstractTypeValidatorBuilder[str]):
    def __init__(self):
        super().__init__(str)

    def length_in_range(self, a: int | None, b: int | None, /) -> StrValidatorBuilder:
        match (a, b):
            case (int(a), None):
                self.check(lambda it: a <= len(it), f'str length less than {a}: "%s"')
            case (None, int(b)):
                self.check(lambda it: len(it) <= b, f'str length over {b}: "%s"')
            case (int(a), int(b)):
                self.check(lambda it: a <= len(it) <= b, f'str length out of range [{a}, {b}]: "%s"')
            case _:
                raise TypeError()
        return self

    def match(self, r: str | re.Pattern) -> StrValidatorBuilder:
        if isinstance(r, str):
            r = re.compile(r)
        self.check(lambda it: r.match(r) is not None, f'str does not match to {r.pattern} : "%s"')
        return self


class IntValidatorBuilder(AbstractTypeValidatorBuilder[int]):
    def __init__(self):
        super().__init__(int)

    def in_range(self, a: int | None, b: int | None, /) -> IntValidatorBuilder:
        match (a, b):
            case (int(a), None):
                self.check(lambda it: a <= it, f'value less than {a}: %d')
            case (None, int(b)):
                self.check(lambda it: it <= b, f'value over {b}: %d')
            case (int(a), int(b)):
                self.check(lambda it: a <= it <= b, f'value out of range [{a}, {b}]: %d')
            case _:
                raise TypeError()

        return self

    def positive(self, include_zero=True):
        if include_zero:
            self.check(lambda it: it > 0)
        else:
            self.check(lambda it: it >= 0)
        return self

    def negative(self, include_zero=True):
        if include_zero:
            self.check(lambda it: it < 0)
        else:
            self.check(lambda it: it <= 0)
        return self


class FloatValidatorBuilder(AbstractTypeValidatorBuilder[float]):
    def __init__(self):
        super().__init__((int, float))
        self.__allow_nan = False

    def in_range(self, a: float | None, b: float | None, /) -> FloatValidatorBuilder:
        match (a, b):
            case (int(a), None):
                self.check(lambda it: a < it, f'value less than {a}: %f')
            case (None, int(b)):
                self.check(lambda it: it < b, f'value over {b}: %f')
            case (int(a), int(b)):
                self.check(lambda it: a < it < b, f'value out of range ({a}, {b}): %f')
            case _:
                raise TypeError()

        return self

    def in_range_closed(self, a: float | None, b: float | None, /) -> FloatValidatorBuilder:
        match (a, b):
            case (int(a), None):
                self.check(lambda it: a <= it, f'value less than {a}: %f')
            case (None, int(b)):
                self.check(lambda it: it <= b, f'value over {b}: %f')
            case (int(a), int(b)):
                self.check(lambda it: a <= it <= b, f'value out of range [{a}, {b}]: %f')
            case _:
                raise TypeError()
        return self

    def allow_nan(self, allow: bool = True) -> FloatValidatorBuilder:
        self.__allow_nan = allow
        return self

    def positive(self, include_zero=True):
        if include_zero:
            self.check(lambda it: it >= 0)
        else:
            self.check(lambda it: it > 0)
        return self

    def negative(self, include_zero=True):
        if include_zero:
            self.check(lambda it: it <= 0)
        else:
            self.check(lambda it: it < 0)
        return self

    def __call__(self, value: Any) -> bool:
        if value != value:
            return self.__allow_nan

        return super().__call__(value)


class ListValidatorBuilder(AbstractTypeValidatorBuilder[list[T]]):
    def __init__(self, element_type: type[T] = None):
        super().__init__()
        self.__element_type = element_type

    def length_in_range(self, a: int | None, b: int | None, /) -> ListValidatorBuilder:
        match (a, b):
            case (int(a), None):
                self.check(lambda it: a <= len(it),
                           lambda it: f'list length less than {a}: {len(it)}')
            case (None, int(b)):
                self.check(lambda it: len(it) <= b,
                           lambda it: f'list length over {b}: {len(it)}')
            case (int(a), int(b)):
                self.check(lambda it: a <= len(it) <= b,
                           lambda it: f'list length out of range [{a}, {b}]: {len(it)}')
            case _:
                raise TypeError()

        return self

    def on_item(self, validator: Callable[[Any], bool]) -> TupleValidatorBuilder:
        self.check(ListItemValidatorBuilder(validator))
        return self

    def __call__(self, value: Any) -> bool:
        if not isinstance(value, (tuple, list)):
            raise ValidatorFailError(f'not a list : {value}')

        if (element_type := self.__element_type) is not None:
            for i, element in enumerate(value):
                if not element_isinstance(element, element_type):
                    raise ValidatorFailError(f'wrong element type at {i} : {element}')

        return super().__call__(value)


class TupleValidatorBuilder(AbstractTypeValidatorBuilder[tuple]):
    def __init__(self, element_type: tuple[type[T], ...]):
        super().__init__()
        self.__element_type = element_type

    def length_in_range(self, a: int | None, b: int | None, /) -> TupleValidatorBuilder:
        match (a, b):
            case (int(a), None):
                self.check(lambda it: a <= len(it),
                           lambda it: f'list length less than {a}: {len(it)}')
            case (None, int(b)):
                self.check(lambda it: len(it) <= b,
                           lambda it: f'list length over {b}: {len(it)}')
            case (int(a), int(b)):
                self.check(lambda it: a <= len(it) <= b,
                           lambda it: f'list length out of range [{a}, {b}]: {len(it)}')
            case _:
                raise TypeError()
        return self

    def on_item(self, item: int, validator: Callable[[Any], bool]) -> TupleValidatorBuilder:
        if item < 0:
            raise ValueError('should always use positive index')

        et = self.__element_type[item]  # check range
        if et is ...:
            raise IndexError()

        self.check(TupleItemValidatorBuilder(item, validator))

        return self

    def __call__(self, value: Any) -> bool:
        if not isinstance(value, tuple):
            raise ValidatorFailError(f'not a tuple : {value}')

        if len(element_type := self.__element_type) > 0:
            if element_type[-1] is ...:
                at_least_length = len(element_type) - 1
                if len(value) < at_least_length:
                    raise ValidatorFailError(f'length less than {at_least_length} : {value}')

                for i, e, t in zip(range(at_least_length), value, element_type):
                    if not element_isinstance(e, t):
                        raise ValidatorFailError(f'wrong element type at {i} : {e}')

                if at_least_length > 0:
                    last_element_type = element_type[at_least_length - 1]
                    for i, e in zip(range(at_least_length, len(value)), value[at_least_length:]):
                        if not element_isinstance(e, last_element_type):
                            raise ValidatorFailError(f'wrong element type at {i} : {e}')

            else:
                if len(value) != len(element_type):
                    raise ValidatorFailError(f'length not match to {len(element_type)} : {value}')

                for i, e, t in zip(range(len(element_type)), value, element_type):
                    if not element_isinstance(e, t):
                        raise ValidatorFailError(f'wrong element type at {i} : {e}')

        return super().__call__(value)


class ListItemValidatorBuilder(LambdaValidator):
    def __call__(self, value: Any) -> bool:
        for i, element in enumerate(value):
            try:
                fail = not super().__call__(element)
            except BaseException as e:
                raise ValidatorFailError(f'at index {i}, ' + e.args[0]) from e
            else:
                if fail:
                    raise ValidatorFailError(f'at index {i}, validate fail : {value}')
        return True


class TupleItemValidatorBuilder(LambdaValidator):
    def __init__(self, item: int, validator: Callable[[Any], bool]):
        super().__init__(validator)
        self.__item = item

    def __call__(self, value: Any) -> bool:
        try:
            element = value[self.__item]
        except IndexError as e:
            raise ValidatorFailError(f'index {self.__item} out of size {len(value)}') from e
        else:
            try:
                return super().__call__(element)
            except BaseException as e:
                raise ValidatorFailError(f'at index {self.__item}, ' + e.args[0]) from e


class OrValidatorBuilder:
    def __init__(self, *validator: Callable[[Any], bool]):
        self.__validators = validator

    def __call__(self, value: Any) -> bool:
        if len(self.__validators) == 0:
            return True

        coll = []
        for validator in self.__validators:
            try:
                if validator(value):
                    return True
            except BaseException as e:
                if len(e.args):
                    coll.append(e.args[0])

        raise ValidatorFailError('; '.join(coll))


class AndValidatorBuilder:
    def __init__(self, *validator: Callable[[Any], bool]):
        self.__validators = validator

    def __call__(self, value: Any) -> bool:
        if len(self.__validators) == 0:
            return True

        for validator in self.__validators:
            if not validator(value):
                raise ValidatorFailError()

        return True


def element_isinstance(e, t) -> bool:
    if isinstance(t, type):
        return isinstance(e, t)

    if t is Any:
        return True

    if callable(t):
        try:
            return True if t(e) else False
        except TypeError:
            return False

    print(f'NotImplementedError(element_isinstance(..., {t}))')
    return False
