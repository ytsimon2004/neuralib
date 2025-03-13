from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, overload, TypeVar, Generic

__all__ = ['validator']

T = TypeVar('T')


class ValidatorFailError(ValueError):
    pass


class Validator:
    def __call__(self, value: Any) -> bool:
        return True


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

    @property
    def tuple(self, *element_type: type[T]) -> TupleValidatorBuilder:
        return TupleValidatorBuilder(element_type)

    @property
    def list(self, element_type: type[T] = None) -> ListValidatorBuilder:
        return ListValidatorBuilder(element_type)

    @classmethod
    def and_(cls, *validator: Callable[[T], bool]) -> AndValidatorBuilder:
        return AndValidatorBuilder(*validator)

    @classmethod
    def or_(cls, *validator: Callable[[T], bool]) -> OrValidatorBuilder:
        return OrValidatorBuilder(*validator)


validator = ValidatorBuilder()


class AbstractTypeValidatorBuilder(Validator, Generic[T]):
    def __init__(self, value_type: type[T] | tuple[type[T], ...] = None):
        self.value_type = value_type
        self.validators: list[tuple[str, Callable[[T], bool]]] = []

    def __call__(self, value: Any) -> bool:
        if self.value_type is not None and not isinstance(value, self.value_type):
            raise ValueError()

        for message, validator in self.validators:
            try:
                fail = not validator(value)
            except BaseException as e:
                raise ValidatorFailError(message % value) from e
            else:
                if fail:
                    raise ValidatorFailError(message % value)

        return True

    def check(self, validator: Callable[[T], bool], message: str = None):
        if message is None:
            message = 'validate fail'

        self.validators.append((message, validator))


class StrValidatorBuilder(AbstractTypeValidatorBuilder[str]):
    def __init__(self):
        super().__init__(str)

    @overload
    def length_in_range(self, min: int, /) -> StrValidatorBuilder:
        pass

    @overload
    def length_in_range(self, min: int | None, max: int, /) -> StrValidatorBuilder:
        pass

    def length_in_range(self, a, b=None, /) -> StrValidatorBuilder:
        match (a, b):
            case (int(a), None):
                self.check(lambda it: a <= len(it), f'str length less than {a}: "%s"')
            case (None, int(b)):
                self.check(lambda it: len(it) <= b, f'str length over {b}: "%s"')
            case (int(a), int(b)):
                self.check(lambda it: a <= len(it) <= b, f'str length out of range [{a}, {b}]: "%s"')
            case _:
                pass

        return self

    def match(self, r: str | re.Pattern) -> StrValidatorBuilder:
        if isinstance(r, str):
            r = re.compile(r)
        self.check(lambda it: r.match(r) is not None, f'str does not match to {r.pattern} : "%s"')
        return self

    def one_of(self, *candidate: str):
        raise NotImplementedError


class IntValidatorBuilder(AbstractTypeValidatorBuilder[int]):
    def __init__(self):
        super().__init__(int)

    @overload
    def in_range(self, min: int, /) -> IntValidatorBuilder:
        pass

    @overload
    def in_range(self, min: int | None, max: int, /) -> IntValidatorBuilder:
        pass

    def in_range(self, a, b=None, /) -> IntValidatorBuilder:
        raise NotImplementedError


class FloatValidatorBuilder(AbstractTypeValidatorBuilder[float]):
    def __init__(self):
        super().__init__((int, float))

    @overload
    def in_range(self, min: int, /) -> FloatValidatorBuilder:
        pass

    @overload
    def in_range(self, min: int | None, max: int, /) -> FloatValidatorBuilder:
        pass

    def in_range(self, a, b=None, /) -> FloatValidatorBuilder:
        raise NotImplementedError

    def allow_nan(self, allow: bool = True) -> FloatValidatorBuilder:
        raise NotImplementedError


class ListValidatorBuilder(AbstractTypeValidatorBuilder[list[T]]):
    def __init__(self, element_type: type[T] = None):
        super().__init__()
        self.element_type = element_type

    @overload
    def length_in_range(self, min: int, /) -> ListValidatorBuilder:
        pass

    @overload
    def length_in_range(self, min: int | None, max: int, /) -> ListValidatorBuilder:
        pass

    def length_in_range(self, a, b=None, /) -> ListValidatorBuilder:
        raise NotImplementedError

    def __call__(self, value: Any) -> bool:
        if not isinstance(value, (tuple, list)):
            raise ValidatorFailError(f'not a list : {value}')

        if (element_type := self.element_type) is not None:
            for i, element in enumerate(value):
                if not element_isinstance(element, element_type):
                    raise ValidatorFailError(f'wrong element type at {i} : {element}')

        return super().__call__(value)


class TupleValidatorBuilder(AbstractTypeValidatorBuilder[tuple]):
    def __init__(self, element_type: tuple[type[T], ...]):
        super().__init__()
        self.element_type = element_type

    @overload
    def length_in_range(self, min: int, /) -> TupleValidatorBuilder:
        pass

    @overload
    def length_in_range(self, min: int | None, max: int, /) -> TupleValidatorBuilder:
        pass

    def length_in_range(self, a, b=None, /) -> TupleValidatorBuilder:
        raise NotImplementedError

    def __call__(self, value: Any) -> bool:
        if not isinstance(value, tuple):
            raise ValidatorFailError(f'not a tuple : {value}')

        if len(element_type := self.element_type) > 0:
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


class OrValidatorBuilder:
    def __init__(self, *validator: Callable[[Any], bool]):
        self.validators = validator

    def __call__(self, value: Any) -> bool:
        if len(self.validators) == 0:
            return True

        coll = []
        for validator in self.validators:
            try:
                if validator(value):
                    return True
            except BaseException as e:
                if len(e.args):
                    coll.append(e.args[0])

        raise ValidatorFailError('; '.join(coll))


class AndValidatorBuilder:
    def __init__(self, *validator: Callable[[Any], bool]):
        self.validators = validator

    def __call__(self, value: Any) -> bool:
        if len(self.validators) == 0:
            return True

        for validator in self.validators:
            if not validator(value):
                raise ValidatorFailError()

        return True


def element_isinstance(e, t) -> bool:
    if isinstance(t, type):
        return isinstance(t, type)

    if t is Any:
        return True

    if callable(t):
        try:
            return True if t(e) else False
        except TypeError:
            return False

    print(f'NotImplementedError(element_isinstance(..., {t}))')
    return False
