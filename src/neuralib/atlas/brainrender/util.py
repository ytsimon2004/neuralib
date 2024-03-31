from __future__ import annotations

from brainrender._colors import get_random_colors

__all__ = ['get_color']


def get_color(i, color_pattern: str | tuple[str, ...] | list[str]) -> str:
    """
    get color
    :param i: idx of the color pattern list
    :param color_pattern: color pattern list or single element str
    :return:
        color name
    """
    if isinstance(color_pattern, (list, tuple)):
        if i >= len(color_pattern):
            color_pattern.extend(get_random_colors(i + 2 - len(color_pattern)))
        return color_pattern[i]
    elif isinstance(color_pattern, str):
        return color_pattern
