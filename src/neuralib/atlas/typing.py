from typing import Literal

from typing_extensions import TypeAlias

__all__ = ['Area', 'Source', 'Channel', 'MergeLevel',
           'HEMISPHERE_TYPE']

Area: TypeAlias = str
"""Brain area"""

Source: TypeAlias = str
"""Name of the injection alias"""

Channel: TypeAlias = str
"""Fluorescence channel"""

MergeLevel: TypeAlias = int
"""Hierarchical tree merge level"""

#
HEMISPHERE_TYPE = Literal['ipsi', 'contra', 'both']
"""brain hemisphere"""
