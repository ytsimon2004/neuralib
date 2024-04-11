"""
Stimpy parser
==============

:author:
    Yu-Ting Wei


This module provide parser utilities for data acquired from stimpy (lab internal usage)


Including
----------

- logs (.log, .riglog, .stimlog, .camlog)
- protocol file (.prot)
- preference file (.prefs)
- reproduce the experimental stimuli (under dev)
- other utilities (synced time, ...)


**Design Only For the branch follows the git history**
--------------------------------------------------------

- pyvstim system master branch (legacy acquisition version)
- stimpy bitbucket master branch
- stimpy github master branch


Example of riglog data
-----------------------------

.. code-block:: python

    from neuralib.stimpy import RiglogData, PyVlog

    root_path = ...  # directory with all the stimpy output
    rig = RiglogData(root_path)  # stimpy
    # rig = PyVlog(root_path)  # if pyvstim

    print(rig.lap_event.time)  # get lick event time
    print(rig.camera_event['facecam'].time)  # get face camera event time




Example of stimlog data
-----------------------------

.. code-block:: python

    stimlog = rig.stimlog_data()

    # stimulation time segment (on-off) in sec (N, 2), already synced time to riglog
    print(stimlog.stimulus_segment)

    # get foreach stimulus generator (index, stimulus_time, sf, tf, ori)
    pattern = stimlog.get_stim_pattern()
    for si, st, sf, tf, dire in pattern.foreach_stimulus():
        print(f'idx:{si}, time:{st}, sf:{sf}, tf:{tf}, dire:{dire}')




Example of protocol
-----------------------------

.. code-block:: python

    prot = rig.get_prot_file()
    print(prot)  # see the protocol content
    print(prot.visual_stimuli_dataframe)  # see the parsed protocol dataframe

"""

from .baselog import *
from .baseprot import *
from .camlog import *
from .event import *
from .session import *
from .stimpy_core import *
from .stimpy_git import *
from .stimpy_pyv import *
from .stimulus import *
from .util import *
