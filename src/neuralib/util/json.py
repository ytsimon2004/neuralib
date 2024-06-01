from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ['JsonEncodeHandler',
           #
           'load_from_json',
           'save_to_json']


class JsonEncodeHandler(json.JSONEncoder):
    """Extend from the JSONEncoder class and handle the conversions in a default method"""

    def default(self, obj: Any):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)

        return json.JSONEncoder.default(self, obj)


def load_from_json(filepath: str | Path) -> dict:
    with open(filepath, "r") as file:
        return json.load(file)


def save_to_json(filepath: str | Path, my_dict: dict) -> None:
    with open(filepath, "w") as outfile:
        json.dump(my_dict, outfile, sort_keys=True, indent=4)
