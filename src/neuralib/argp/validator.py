"""
Validator Usage Guide
=====================

Overview
--------
This guide demonstrates how to use the validator builders provided by our library. Each section
provides short examples followed by a reference table of the builder’s methods.

.. contents::
   :local:
   :depth: 2


String Validation
=================

Examples
--------
**Minimum String Length**::

    from neuralib.argp import validator

    class Opt:
        # Must be at least 2 characters long
        a: str = argument('-a', validator.str.length_in_range(2, None))

    opt = Opt()
    opt.a = 'Hi'    # OK
    opt.a = ''      # Raises ValueError

**Regex Matching**::

    class Opt:
        # Must match a letter followed by a digit, e.g. 'a1', 'b9'
        a: str = argument('-a', validator.str.match(r'[a-z][0-9]'))

    opt = Opt()
    opt.a = 'a1'    # OK
    opt.a = 'A1'    # Raises ValueError

Method Reference
----------------
.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - **Method**
     - **Description**
   * - :meth:`length_in_range(a, b) <StrValidatorBuilder.length_in_range>`
     - Enforces a string length in [a, b]. Either bound may be ``None``.
   * - :meth:`match(pattern) <StrValidatorBuilder.match>`
     - Checks if the string matches a given regex pattern.
   * - :meth:`starts_with(prefix) <StrValidatorBuilder.starts_with>`
     - Checks if the string starts with ``prefix``.
   * - :meth:`ends_with(suffix) <StrValidatorBuilder.ends_with>`
     - Checks if the string ends with ``suffix``.
   * - :meth:`contains(substring) <StrValidatorBuilder.contains>`
     - Checks if the string contains the given substring.
   * - :meth:`is_in(options) <StrValidatorBuilder.is_in>`
     - Checks if the string is in the provided collection of allowed options.


Integer Validation
==================

Examples
--------
**Integer Range**::

    class Opt:
        # Must be >= 2
        a: int = argument('-a', validator.int.in_range(2, None))

    opt = Opt()
    opt.a = 5   # OK
    opt.a = 0   # Raises ValueError

**Positivity**::

    class Opt:
        # Must be strictly positive
        a: int = argument('-a', validator.int.positive(include_zero=False))

    opt = Opt()
    opt.a = 10  # OK
    opt.a = 0   # Raises ValueError

Method Reference
----------------
.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - **Method**
     - **Description**
   * - :meth:`in_range(a, b) <IntValidatorBuilder.in_range>`
     - Checks if integer is in [a, b]. Either bound may be ``None``.
   * - :meth:`positive(include_zero=True) <IntValidatorBuilder.positive>`
     - Checks if integer is >= 0 (if ``include_zero=True``) or > 0 otherwise.
   * - :meth:`negative(include_zero=True) <IntValidatorBuilder.negative>`
     - Checks if integer is <= 0 (if ``include_zero=True``) or < 0 otherwise.


Float Validation
================

Examples
--------
**Range + NaN Handling**::

    class Opt:
        # Must be < 100, NaN not allowed
        a: float = argument('-a',
            validator.float.in_range(None, 100).allow_nan(False)
        )

    opt = Opt()
    opt.a = 3.14        # OK
    opt.a = 123.45      # Raises ValueError (out of range)
    opt.a = float('nan')# Raises ValueError (NaN not allowed)

Method Reference
----------------
.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - **Method**
     - **Description**
   * - :meth:`in_range(a, b) <FloatValidatorBuilder.in_range>`
     - Checks if float is in the open interval ``(a, b)``.
   * - :meth:`in_range_closed(a, b) <FloatValidatorBuilder.in_range_closed>`
     - Checks if float is in the closed interval ``[a, b]``.
   * - :meth:`allow_nan(allow=True) <FloatValidatorBuilder.allow_nan>`
     - Allows or disallows NaN values.
   * - :meth:`positive(include_zero=True) <FloatValidatorBuilder.positive>`
     - Checks if float is >= 0 (if ``include_zero=True``) or > 0 otherwise.
   * - :meth:`negative(include_zero=True) <FloatValidatorBuilder.negative>`
     - Checks if float is <= 0 (if ``include_zero=True``) or < 0 otherwise.


List Validation
===============

Examples
--------
**List of Integers**::

    class Opt:
        # Must be a list of integers
        a: list[int] = argument('-a', validator.list(int))

    opt = Opt()
    opt.a = [1, 2, 3]    # OK
    opt.a = ['a', 2]     # Raises ValueError

**Item Validation**::

    class Opt:
        # Each item must be non-negative
        a: list[int] = argument('-a',
            validator.list(int).on_item(validator.int.positive(True))
        )

    opt = Opt()
    opt.a = [0, 2, 5]    # OK
    opt.a = [1, -1]      # Raises ValueError

Method Reference
----------------
.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - **Method**
     - **Description**
   * - :meth:`length_in_range(a, b) <ListValidatorBuilder.length_in_range>`
     - Enforces list length in [a, b].
   * - :meth:`allow_empty(allow=True) <ListValidatorBuilder.allow_empty>`
     - Allows or disallows an empty list.
   * - :meth:`on_item(validator) <ListValidatorBuilder.on_item>`
     - Applies a validator to each list item.


Tuple Validation
================

Examples
--------
**Fixed-Length Tuple**::

    class Opt:
        # Must be (str, int, float)
        a: tuple[str, int, float] = argument(
            '-a', validator.tuple(str, int, float)
        )

    opt = Opt()
    opt.a = ('abc', 42, 3.14)   # OK
    opt.a = ('abc', 42)        # Raises ValueError (too few elements)

**Variable-Length**::

    class Opt:
        # Must be (str, int, ...) i.e. at least 'str + int', optionally more ints
        a: tuple[str, int, ...] = argument(
            '-a', validator.tuple(str, int, ...)
        )

    opt = Opt()
    opt.a = ('x', 10)            # OK
    opt.a = ('x', 10, 20, 30)    # OK
    opt.a = ('x',)               # Raises ValueError (missing int)

**Item-Validation**::

    class Opt:
        # Must be (str, int, float).
        # The string must have a length <= 5,
        # and the int must be >= 0 and <= 100.
        a: tuple[str, int, float] = argument(
            '-a',
            validator.tuple(str, int, float)
                .on_item(0, validator.str.length_in_range(None, 5))
                .on_item(1, validator.int.in_range(0, 100))
        )

    opt = Opt()

    # Passes all checks: str length=3, int in range [0..100], float is fine
    opt.a = ('hey', 42, 3.14)

    # Fails because the string is too long:
    opt.a = ('excessive', 42, 1.2)
    # Raises ValueError: str length over 5: "excessive"

    # Fails because integer is out of range:
    opt.a = ('hi', 999, 2.5)
    # Raises ValueError: value out of range [0, 100]: 999

Method Reference
----------------
.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - **Method**
     - **Description**
   * - :meth:`on_item(indexes, validator) <TupleValidatorBuilder.on_item>`
     - Apply a validator to specific tuple positions, or ``None`` for all.
   * - *(constructor)*
     - Pass one int (e.g. 3) to enforce a fixed-length tuple with no type checks, or a tuple of types
       like ``(str, int, float)``. The last type can be ``...`` for variable length.


Logical Combinators
===================

Examples
--------
**OR Combination**::

    class Opt:
        # Must be int in [0,10] OR str length in [0,10]
        a: int | str = argument(
            '-a',
            validator.any(
                validator.int.in_range(0, 10),
                validator.str.length_in_range(0, 10)
            )
        )

    opt = Opt()
    opt.a = 5            # OK (int in [0..10])
    opt.a = 'abc'        # OK (length=3)
    opt.a = 50           # Raises ValueError

**AND Combination**::

    class Opt:
        # Must be non-negative AND non-positive => zero
        a: int = argument('-a', validator.all(
            validator.int.positive(include_zero=True),
            validator.int.negative(include_zero=True)
        ))

    opt = Opt()
    opt.a = 0   # OK
    opt.a = 1   # Raises ValueError
    opt.a = -1  # Raises ValueError

Method Reference
----------------
.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - **Method/Class**
     - **Description**
   * - :meth:`validator.any(...) <ValidatorBuilder.any>` or ``|``
     - Combine validators with logical OR; passing at least one is enough.
   * - :meth:`validator.all(...) <ValidatorBuilder.all>` or ``&``
     - Combine validators with logical AND; must pass them all.
   * - ``OrValidatorBuilder``
     - The class implementing OR logic.
   * - ``AndValidatorBuilder``
     - The class implementing AND logic.


Error Handling
--------------
If any validation fails:
- A :class:`ValidatorFailError` (or subclass) is raised, often rethrown as ``ValueError``
  in higher-level frameworks.
- **Type mismatches** specifically raise :class:`ValidatorFailOnTypeError`.


"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, TypeVar, Generic, final, overload, Collection

from typing_extensions import Self

T = TypeVar('T')


class ValidatorFailError(ValueError):
    pass


class ValidatorFailOnTypeError(ValidatorFailError):
    """
    A special ValidatorFailError that is raised when type validation failure.
    It is used for Validator#any() to exclude some error message.
    """
    pass


class Validator:
    def __call__(self, value: Any) -> bool:
        return True


class LambdaValidator(Validator, Generic[T]):
    def __init__(self, validator: Callable[[T], bool],
                 message: str | Callable[[T], str] = None):
        if isinstance(message, str):
            message = message.__mod__

        self.__validator = validator
        self.__message = message

    def __call__(self, value: T) -> bool:
        message = self.__message
        try:
            success = self.__validator(value)
        except ValidatorFailError:
            raise
        except BaseException as e:
            if message is None:
                raise ValidatorFailError('validate failure') from e
            else:
                raise ValidatorFailError(message(value)) from e
        else:
            if success is None or success:
                return True
            elif message is None:
                return False
            else:
                raise ValidatorFailError(message(value))

    def __and__(self, validator: Callable[[Any], bool]) -> AndValidatorBuilder:
        return AndValidatorBuilder(self, validator)

    def __or__(self, validator: Callable[[Any], bool]) -> OrValidatorBuilder:
        return OrValidatorBuilder(self, validator)


@final
class ValidatorBuilder:
    @property
    def str(self) -> StrValidatorBuilder:
        return StrValidatorBuilder()

    @property
    def int(self) -> IntValidatorBuilder:
        return IntValidatorBuilder()

    @property
    def float(self) -> FloatValidatorBuilder:
        return FloatValidatorBuilder()

    @overload
    def tuple(self, element_type: int) -> TupleValidatorBuilder:
        pass

    @overload
    def tuple(self, *element_type: type[T]) -> TupleValidatorBuilder:
        pass

    # noinspection PyMethodMayBeStatic
    def tuple(self, *element_type) -> TupleValidatorBuilder:
        return TupleValidatorBuilder(element_type)

    # noinspection PyMethodMayBeStatic
    def list(self, element_type: type[T] = None) -> ListValidatorBuilder:
        return ListValidatorBuilder(element_type)

    @classmethod
    def all(cls, *validator: Callable[[T], bool]) -> AndValidatorBuilder:
        return AndValidatorBuilder(*validator)

    @classmethod
    def any(cls, *validator: Callable[[T], bool]) -> OrValidatorBuilder:
        return OrValidatorBuilder(*validator)

    @classmethod
    def optional(cls) -> Validator:
        return LambdaValidator(lambda it: it is None)

    @classmethod
    def non_none(cls) -> Validator:
        return LambdaValidator(lambda it: it is not None)

    def __call__(self, validator: Callable[[Any], bool],
                 message: str | Callable[[Any], str] = None) -> LambdaValidator:
        return LambdaValidator(validator, message)


class AbstractTypeValidatorBuilder(Validator, Generic[T]):
    def __init__(self, value_type: type[T] | tuple[type[T], ...] = None):
        self.__value_type = value_type
        self.__validators: list[LambdaValidator[T]] = []
        self.__allow_none = False

    def __call__(self, value: Any) -> bool:
        if value is None:
            if self.__allow_none:
                return True
            else:
                raise ValidatorFailError('None')

        # noinspection PyTypeHints
        if self.__value_type is not None and not isinstance(value, self.__value_type):
            raise ValidatorFailOnTypeError(f'not instance of {self.__value_type.__name__} : {value}')

        for validator in self.__validators:
            if not validator(value):
                return False

        return True

    @overload
    def _add(self, validator: LambdaValidator[T]):
        pass

    @overload
    def _add(self, validator: Callable[[T], bool], message: str | Callable[[T], str] = None):
        pass

    def _add(self, validator, message=None):
        if not isinstance(validator, LambdaValidator):
            validator = LambdaValidator(validator, message)
        self.__validators.append(validator)

    def optional(self) -> Self:
        self.__allow_none = True
        return self

    def __and__(self, validator: Callable[[Any], bool]) -> AndValidatorBuilder:
        return AndValidatorBuilder(self, validator)

    def __or__(self, validator: Callable[[Any], bool]) -> OrValidatorBuilder:
        return OrValidatorBuilder(self, validator)


class StrValidatorBuilder(AbstractTypeValidatorBuilder[str]):
    def __init__(self):
        super().__init__(str)

    def length_in_range(self, a: int | None, b: int | None, /) -> StrValidatorBuilder:
        """Enforce a string length range"""
        match (a, b):
            case (int(a), None):
                self._add(lambda it: a <= len(it), f'str length less than {a}: "%s"')
            case (None, int(b)):
                self._add(lambda it: len(it) <= b, f'str length over {b}: "%s"')
            case (int(a), int(b)):
                self._add(lambda it: a <= len(it) <= b, f'str length out of range [{a}, {b}]: "%s"')
            case _:
                raise TypeError()
        return self

    def match(self, r: str | re.Pattern) -> StrValidatorBuilder:
        """Check if string matches a regular expression"""
        if isinstance(r, str):
            r = re.compile(r)
        self._add(lambda it: r.match(it) is not None, f'str does not match to {r.pattern} : "%s"')
        return self

    def starts_with(self, prefix: str) -> StrValidatorBuilder:
        """Check if string values start with a substring"""
        self._add(lambda it: it.startswith(prefix), f'str does not start with {prefix}: "%s"')
        return self

    def ends_with(self, suffix: str) -> StrValidatorBuilder:
        """Check if string values end with a substring"""
        self._add(lambda it: it.endswith(suffix), f'str does not end with {suffix}: "%s"')
        return self

    def contains(self, text: str) -> StrValidatorBuilder:
        """Check if string values contain a substring"""
        self._add(lambda it: text in it, f'str does not contain {text}: "%s"')
        return self

    def is_in(self, options: Collection[str]) -> StrValidatorBuilder:
        """Check if string is one of the allow options"""
        self._add(lambda it: it in options, f'str not in allowed set {options}: "%s"')
        return self


class IntValidatorBuilder(AbstractTypeValidatorBuilder[int]):
    def __init__(self):
        super().__init__(int)

    def in_range(self, a: int | None, b: int | None, /) -> IntValidatorBuilder:
        """Enforce a numeric range for int values"""
        match (a, b):
            case (int(a), None):
                self._add(lambda it: a <= it, f'value less than {a}: %d')
            case (None, int(b)):
                self._add(lambda it: it <= b, f'value over {b}: %d')
            case (int(a), int(b)):
                self._add(lambda it: a <= it <= b, f'value out of range [{a}, {b}]: %d')
            case _:
                raise TypeError()

        return self

    def positive(self, include_zero=True):
        """Check if an int value is positive or non-negative"""
        if include_zero:
            self._add(lambda it: it >= 0, 'not a non-negative value : %d')
        else:
            self._add(lambda it: it > 0, 'not a positive value : %d')
        return self

    def negative(self, include_zero=True):
        """Check if an int value is negative or non-positive."""
        if include_zero:
            self._add(lambda it: it <= 0, 'not a non-positive value : %d')
        else:
            self._add(lambda it: it < 0, 'not a negative value : %d')
        return self


class FloatValidatorBuilder(AbstractTypeValidatorBuilder[float]):
    def __init__(self):
        super().__init__((int, float))
        self.__allow_nan = False

    def in_range(self, a: float | None, b: float | None, /) -> FloatValidatorBuilder:
        """Enforce an open-interval numeric range (a < value < b)"""
        match (a, b):
            case (int(a), None):
                self._add(lambda it: a < it, f'value less than {a}: %f')
            case (None, int(b)):
                self._add(lambda it: it < b, f'value over {b}: %f')
            case (int(a), int(b)):
                self._add(lambda it: a < it < b, f'value out of range ({a}, {b}): %f')
            case _:
                raise TypeError()

        return self

    def in_range_closed(self, a: float | None, b: float | None, /) -> FloatValidatorBuilder:
        """ Enforce a closed-interval numeric range (a <= value <= b)"""
        match (a, b):
            case (int(a), None):
                self._add(lambda it: a <= it, f'value less than {a}: %f')
            case (None, int(b)):
                self._add(lambda it: it <= b, f'value over {b}: %f')
            case (int(a), int(b)):
                self._add(lambda it: a <= it <= b, f'value out of range [{a}, {b}]: %f')
            case _:
                raise TypeError()
        return self

    def allow_nan(self, allow: bool = True) -> FloatValidatorBuilder:
        """Allow or disallow NaN (not a number) as a valid float"""
        self.__allow_nan = allow
        return self

    def positive(self, include_zero=True) -> FloatValidatorBuilder:
        """Check if a float value is positive or non-negative"""
        if include_zero:
            self._add(lambda it: it >= 0, 'not a non-negative value : %f')
        else:
            self._add(lambda it: it > 0, 'not a positive value: %f')
        return self

    def negative(self, include_zero=True) -> FloatValidatorBuilder:
        """Check if a float value is negative or non-positive"""
        if include_zero:
            self._add(lambda it: it <= 0, 'not a non-positive value : %f')
        else:
            self._add(lambda it: it < 0, 'not a negative value : %f')
        return self

    def __call__(self, value: Any) -> bool:
        if value != value:
            if self.__allow_nan:
                return True
            else:
                raise ValidatorFailError('NaN')

        return super().__call__(value)


class ListValidatorBuilder(AbstractTypeValidatorBuilder[list[T]]):
    def __init__(self, element_type: type[T] = None):
        super().__init__()
        self.__element_type = element_type
        self.__allow_empty = True

    def length_in_range(self, a: int | None, b: int | None, /) -> ListValidatorBuilder:
        """Enforce a length range for lists"""
        match (a, b):
            case (int(a), None):
                self._add(lambda it: a <= len(it),
                          lambda it: f'list length less than {a}: {len(it)}')
            case (None, int(b)):
                self._add(lambda it: len(it) <= b,
                          lambda it: f'list length over {b}: {len(it)}')
            case (int(a), int(b)):
                self._add(lambda it: a <= len(it) <= b,
                          lambda it: f'list length out of range [{a}, {b}]: {len(it)}')
            case _:
                raise TypeError()

        return self

    def allow_empty(self, allow: bool = True):
        """Allow or disallow empty lists"""
        self.__allow_empty = allow

    def on_item(self, validator: Callable[[Any], bool]) -> ListValidatorBuilder:
        """Apply an additional validator to each item in the list

        :param validator: A callable that validates each item
        """
        self._add(ListItemValidatorBuilder(validator))
        return self

    def __call__(self, value: Any) -> bool:
        if not isinstance(value, (tuple, list)):
            raise ValidatorFailOnTypeError(f'not a list : {value}')

        if not self.__allow_empty and len(value) == 0:
            raise ValidatorFailError(f'empty list : {value}')

        if (element_type := self.__element_type) is not None:
            for i, element in enumerate(value):
                if not element_isinstance(element, element_type):
                    raise ValidatorFailError(f'wrong element type at {i} : {element}')

        return super().__call__(value)


class TupleValidatorBuilder(AbstractTypeValidatorBuilder[tuple]):
    def __init__(self, element_type: tuple[int] | tuple[type[T], ...]):
        super().__init__()

        match element_type:
            case (int(length), ):
                element_type = (None,) * length

        self.__element_type = element_type

    def on_item(self, item: int | list[int] | None, validator: Callable[[Any], bool]) -> TupleValidatorBuilder:
        """Apply a validator to specific tuple positions

        :param item: A single index, a list of indices, or None for all indices
        :param validator: The validation callable to apply
        """
        if item is None:
            pass
        elif isinstance(item, int):
            if item < 0:
                raise ValueError('should always use positive index')
        else:
            for index in item:
                if index < 0:
                    raise ValueError('should always use positive index')

        # check range
        if item is not None and len(self.__element_type) > 0:
            if isinstance(item, int):
                et = self.__element_type[item]
                if et is ...:
                    raise IndexError()
            else:
                for index in item:
                    et = self.__element_type[index]
                    if et is ...:
                        raise IndexError()

        self._add(TupleItemValidatorBuilder(item, validator))

        return self

    def __call__(self, value: Any) -> bool:
        if not isinstance(value, tuple):
            raise ValidatorFailOnTypeError(f'not a tuple : {value}')

        if len(element_type := self.__element_type) > 0:
            if element_type[-1] is ...:
                at_least_length = len(element_type) - 1
                if len(value) < at_least_length:
                    raise ValidatorFailError(f'length less than {at_least_length} : {value}')

                for i, e, t in zip(range(at_least_length), value, element_type):
                    if t is not None and not element_isinstance(e, t):
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
                    if t is not None and not element_isinstance(e, t):
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
    def __init__(self, item: int | list[int] | None, validator: Callable[[Any], bool]):
        super().__init__(validator)
        self.__item = item

    def __call__(self, value: Any) -> bool:
        if self.__item is None:
            for index in range(len(value)):
                if not self.__call_on_index__(index, value):
                    return False
            return True
        elif isinstance(self.__item, int):
            return self.__call_on_index__(self.__item, value)
        else:
            for index in self.__item:
                if not self.__call_on_index__(index, value):
                    return False
            return True

    def __call_on_index__(self, index: int, value: Any) -> bool:
        try:
            element = value[index]
        except IndexError as e:
            raise ValidatorFailError(f'index {index} out of size {len(value)}') from e

        try:
            return super().__call__(element)
        except BaseException as e:
            raise ValidatorFailError(f'at index {index}, ' + e.args[0]) from e


class OrValidatorBuilder(Validator):
    def __init__(self, *validator: Callable[[Any], bool]):
        self.__validators = list(validator)

    def __call__(self, value: Any) -> bool:
        if len(self.__validators) == 0:
            return True

        coll = []
        for validator in self.__validators:
            try:
                if validator(value):
                    return True
            except ValidatorFailOnTypeError:
                pass
            except BaseException as e:
                if len(e.args):
                    coll.append(e.args[0])

        raise ValidatorFailError('; '.join(coll))

    def __and__(self, validator: Callable[[Any], bool]) -> AndValidatorBuilder:
        return AndValidatorBuilder(self, validator)

    def __or__(self, validator: Callable[[Any], bool]) -> OrValidatorBuilder:
        if isinstance(validator, OrValidatorBuilder):
            self.__validators.extend(validator.__validators)
        else:
            self.__validators.append(validator)
        return self


class AndValidatorBuilder(Validator):
    def __init__(self, *validator: Callable[[Any], bool]):
        self.__validators = list(validator)

    def __call__(self, value: Any) -> bool:
        if len(self.__validators) == 0:
            return True

        for validator in self.__validators:
            if not validator(value):
                raise ValidatorFailError()

        return True

    def __and__(self, validator: Callable[[Any], bool]) -> AndValidatorBuilder:
        if isinstance(validator, AndValidatorBuilder):
            self.__validators.extend(validator.__validators)
        else:
            self.__validators.append(validator)
        return self

    def __or__(self, validator: Callable[[Any], bool]) -> OrValidatorBuilder:
        return OrValidatorBuilder(self, validator)


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
