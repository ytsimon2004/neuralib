from typing import TypeAlias, Literal

__all__ = ['Area', 'Source', 'Channel', 'HEMISPHERE_TYPE']

Area: TypeAlias = str
Source: TypeAlias = str
Channel: TypeAlias = str
HEMISPHERE_TYPE = Literal['ipsi', 'contra', 'both']
