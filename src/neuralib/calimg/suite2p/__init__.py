"""
Suite2p Result Parser
=================

:author:
    Yu-Ting Wei

This module provide the usage for 2-photons calcium imaging after suite2p registration/segmentation.

>>> s2p_dir = ""  # suite2p base directory (*/suite2p/plane*)
>>> s2p = Suite2PResult.load(s2p_dir, cell_prob=0.0, channel=0)
>>> print(dir(s2p))  # see available attributes/properties/methods

"""
from .core import *
from .signals import *
