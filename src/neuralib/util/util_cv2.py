from __future__ import annotations

import sys
from typing import TypedDict

__all__ = [
    'KeyMapping',
    #
    'get_keymapping',
    'find_key_from_value'
]


class KeyMapping(TypedDict, total=False):
    """For controlling the keyboard in different OS"""
    escape: int
    backspace: int
    # space: int  # labeller specific printable for notes
    enter: int

    left_square_bracket: int  # [
    right_square_bracket: int  # ]

    left: int
    right: int
    up: int
    down: int

    plus: int  # +
    minus: int  # -
    equal: int  # =


#
COMMON_KEYMAPPING: KeyMapping = {
    'escape': 27,
    # 'space': 32,
    'enter': 13,

    'left_square_bracket': 91,
    'right_square_bracket': 93,

    # ord
    'plus': ord('+'),
    'minus': ord('-'),
    'equal': ord('=')
}

WIN_KEYMAPPING: KeyMapping = {
    **COMMON_KEYMAPPING,
    'backspace': 8,
    'left': 2424832,
    'right': 2555904,
    'up': 2490368,
    'down': 2621440
}

MAC_KEYMAPPING: KeyMapping = {
    **COMMON_KEYMAPPING,
    'backspace': 127,
    'left': 2,
    'right': 3,
    'up': 0,
    'down': 1

}

LINUX_KEYMAPPING: KeyMapping = {
    **COMMON_KEYMAPPING,
    'backspace': 8,
    'left': 81,
    'right': 83,
    'up': 82,
    'down': 84
}


def get_keymapping() -> KeyMapping:
    p = sys.platform
    if p in ('linux', 'linux2'):
        return LINUX_KEYMAPPING
    elif p == 'darwin':
        return MAC_KEYMAPPING
    elif p == 'win32':
        return WIN_KEYMAPPING


def find_key_from_value(dy: KeyMapping, value: int) -> str | bool | None:
    ret = []
    for k, v in dy.items():
        if v == value:
            ret.append(k)

    if len(ret) == 0:
        return
    elif len(ret) == 1:
        return ret[0]
    else:
        raise False
