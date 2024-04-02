"""
Persistence Class for argp supporting
=====================================

:author:
    Ta-Shun Su

"""
import abc
from pathlib import Path
from typing import Generic, TypeVar, get_origin, get_args

from neuralib.argp import argument, copy_argument
from neuralib.persistence import (
    persistence,
    PickleHandler,
    PersistenceHandler,
    AutoIncFieldNotResolvedError
)
from neuralib.util.util_verbose import fprint, print_load, print_save

__all__ = ['PersistenceOptions',
           'get_options_and_cache']

T = TypeVar('T')


def persistence_filename(cache: T) -> str:
    return persistence.filename(cache) + '.pkl'


class PersistenceOptions(Generic[T], metaclass=abc.ABCMeta):
    """The Option class that handle one kind of cache class T, including
    load cache, create cache, and save cache.
    """
    GROUP_CACHE = 'Persistence options'

    invalid_cache: bool = argument(
        '--invalid-cache',
        group=GROUP_CACHE,
        help='invalid persistence data'
    )

    @property
    def persistence_class(self) -> type[T]:
        # https://stackoverflow.com/a/50101934
        for t in type(self).__orig_bases__:
            if get_origin(t) == PersistenceOptions:
                return get_args(t)[0]
        raise TypeError('unable to retrieve cache class T')

    def cache_path(self, dest: Path = None) -> Path:
        return dest

    def persistence_handler(self, dest: Path = None) -> PersistenceHandler[T]:
        return PickleHandler(self.persistence_class, self.cache_path(dest))

    @abc.abstractmethod
    def empty_cache(self) -> T:
        """create an empty cache which only initialize required fields.

        :return: cache instance
        """
        pass

    def find_cache(self, cache: T, dest: Path = None, validator=False) -> list[T]:
        """Find the persistence.

        for all fields

        >>> template = self.empty_cache()
        >>> template.a = 1  # want to find all cache whose `a` equals to 1
        >>> template.b = field_missing # want to find all cache and don't matter what `b` is
        >>> found = self.find_cache(template)

        :param cache:
        :param validator:
        :return:
        """
        handler = self.persistence_handler(dest)

        ret = []

        for file, found in handler.load_all(cache):
            if validator:
                if not self.validate_cache(file, found):
                    continue

            ret.append(found)

        return ret

    def save_cache(self, cache: T, dest: Path = None, force=True):
        save_path = self.cache_path(dest) / persistence_filename(cache)
        if save_path.exists() and not force:
            raise FileExistsError(str(save_path))

        save_path.parent.mkdir(parents=True, exist_ok=True)
        persistence.save(cache, save_path)

    def load_cache(self,
                   cache: T = None,
                   error_when_missing=False,
                   dest: Path = None,
                   **kwargs) -> T:
        """load persistence from disk according to *result*'s required fields.

        :param cache: cache object with necessary fields filled.
        :param error_when_missing: do not try to generate the cache when cache missing.
        :param dest:
        :param kwargs: overwrite field value in *result*.
        :return: cache instance.
        :raise FileNotFoundError: error_when_missing and file not found.
        """
        do_load = not self.invalid_cache
        handler = self.persistence_handler(dest)

        if cache is None:
            cache = self.empty_cache()

        # load/save path
        try:
            output_file = handler.filepath(cache, **kwargs)
        except AutoIncFieldNotResolvedError as e:
            if error_when_missing:
                raise FileNotFoundError from e
            do_load = False
            output_file = None

        if do_load:
            try:
                print_load(output_file)
                cache = handler.load_persistence(output_file)
                if self.validate_cache(output_file, cache):
                    return cache
            except FileNotFoundError:
                if error_when_missing:
                    raise
            except TypeError as e:
                fprint(repr(e), vtype='error')

        elif error_when_missing:
            raise FileNotFoundError(output_file)

        cache = self.compute_cache(cache)
        handler.save_persistence(cache, output_file)
        output_file = handler.filepath(cache)
        print_save(output_file)

        return cache

    def validate_cache(self, cache_path: Path, cache: T) -> bool:
        """Validating loaded cache instance.

        Once validating fail (return False), goto :meth:`_compute_cache`.

        :param cache_path:
        :param cache: loaded cache instance
        :return: False if validating fail.
        """
        return True

    @abc.abstractmethod
    def compute_cache(self, cache: T) -> T:
        """Compute cache according to *cache*'s required fields.

        :param cache: cached instance
        :return: computed cached instance
        """
        pass


def get_options_and_cache(opt_cls: type[PersistenceOptions[T]],
                          ref,
                          error_when_missing=False,
                          **kwargs) -> T:
    """
    copy the arguments from PersistenceOpt (class that compute the cache) to Apply*Opt
    Can be used for analysis apply two different cached files

     **Example**
        >>> @persistence.persistence_class
        >>> class SortIdxCache:
        ...     # for cache attributes declare
        ...     pass

        >>> class SortIdxOptions(PersistenceOptions[SortIdxCache]):
        ...     # for cache computing
        ...     def empty_cache(self) -> SortIdxCache:
        ...         pass
        ...     def _compute_cache(self, cache: SortIdxCache) -> SortIdxCache:
        ...         pass

        >>> class ApplySortIdxOptions:
        ...     # for cache applying, suppose to be the one layer before parent class for analysis class
        ...     def apply_sort_idx_cache(self) -> SortIdxCache:
        ...         return get_options_and_cache(SortIdxOptions, self)

        >>> class CPBeltSortOptions(ApplyPosBinActOptions, ApplySortIdxOptions):
        ...     pass

    :param opt_cls: PersistenceOpt
    :param ref:
    :param error_when_missing:
    :return:
    """
    return copy_argument(opt_cls(), ref).load_cache(error_when_missing=error_when_missing, **kwargs)
