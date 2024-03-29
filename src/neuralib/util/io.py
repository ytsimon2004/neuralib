from pathlib import Path

__all__ = ['CACHE_DIRECTORY',
           'NEUROLIB_CACHE_DIRECTORY',
           'ATLAS_CACHE_DIRECTORY']

CACHE_DIRECTORY = Path.home() / '.cache'
NEUROLIB_CACHE_DIRECTORY = CACHE_DIRECTORY / 'neuralib'

#
ATLAS_CACHE_DIRECTORY = NEUROLIB_CACHE_DIRECTORY / 'atlas'

