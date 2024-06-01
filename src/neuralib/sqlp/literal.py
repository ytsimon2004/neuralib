from typing import Literal

__all__ = ['CONFLICT_POLICY', 'FOREIGN_POLICY', 'UPDATE_POLICY']

CONFLICT_POLICY = Literal['rollback', 'abort', 'fail', 'ignore', 'replace']
""""""
FOREIGN_POLICY = Literal['SET NULL', 'SET DEFAULT', 'CASCADE', 'RESTRICT', 'NO ACTION']
""""""
UPDATE_POLICY = Literal['ABORT', 'FAIL', 'IGNORE', 'REPLACE', 'ROLLBACK']
""""""
