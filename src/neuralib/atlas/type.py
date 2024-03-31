from typing import Literal

from typing_extensions import TypeAlias

__all__ = ['Area', 'Source', 'Channel', 'MergeLevel',
           'HEMISPHERE_TYPE']

Area: TypeAlias = str
Source: TypeAlias = str
Channel: TypeAlias = str
MergeLevel: TypeAlias = int

#
HEMISPHERE_TYPE = Literal['ipsi', 'contra', 'both']
