from typing import Literal

__all__ = ['Area', 'Source', 'Channel', 'TreeLevel',
           'HEMISPHERE_TYPE', 'PLANE_TYPE']

Area = str
"""Brain area"""

Source = str
"""Name of the injection alias"""

Channel = str
"""Fluorescence channel"""

TreeLevel = int
"""Hierarchical tree merge level"""

HEMISPHERE_TYPE = Literal['ipsi', 'contra', 'both']
"""brain hemisphere"""

PLANE_TYPE = Literal['coronal', 'sagittal', 'transverse']
"""brain section (plane) type"""
