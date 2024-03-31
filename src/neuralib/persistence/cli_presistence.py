"""
Persistence Class for argp supporting
=====================================

:author:
    Ta-Shun Su

"""
import abc
from typing import Generic, TypeVar

__all__ = ['PersistenceOptions']

T = TypeVar('T')


# TODO
class PersistenceOptions(Generic[T], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def empty_cache(self) -> T:
        pass

    def validate_cache(self, result: T) -> bool:
        return True

    @abc.abstractmethod
    def compute_cache(self, result: T) -> T:
        pass
