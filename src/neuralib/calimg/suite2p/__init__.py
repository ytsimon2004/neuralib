"""
Suite2p Result Parser & Utils
===============================

:author:
    Yu-Ting Wei

This module provide the container for suite2p registration/segmentation 2P results.

Also include utilities for signal processing and basic plotting

.. code-block:: python

    s2p_dir = ""  # suite2p base directory (*/suite2p/plane*)
    s2p = Suite2PResult.load(s2p_dir, cell_prob=0.0, channel=0)

See available attributes/properties/methods
--------------------------------------------
.. code-block:: python

    print(dir(s2p))


See the option config for running the GUI
---------------------------------------------

.. code-block:: python

    from rich.pretty import pprint
    pprint(s2p.ops)

"""
from .core import *
from .plot import *
from .signals import *
