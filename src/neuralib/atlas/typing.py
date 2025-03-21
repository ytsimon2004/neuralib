from typing import Literal

__all__ = ['Area', 'Source', 'Channel', 'MergeLevel',
           'HEMISPHERE_TYPE']

Area = str
"""Brain area"""

Source = str
"""Name of the injection alias"""

Channel = str
"""Fluorescence channel"""

MergeLevel = int
"""Hierarchical tree merge level"""

HEMISPHERE_TYPE = Literal['ipsi', 'contra', 'both']
"""brain hemisphere"""
