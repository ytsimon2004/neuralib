
import json
from pathlib import Path
from typing import Any

import numpy as np
from neuralib.typing import PathLike
from neuralib.util.verbose import print_save, print_load

__all__ = ['JsonEncodeHandler',
           'load_json',
           'save_json']


class JsonEncodeHandler(json.JSONEncoder):
    """Extend from the JSONEncoder class and handle the conversions in a default method

    **Usage**: Add kwarg, e.g., ``json_dump(..., cls=JsonEncodeHandler)``
    """

    def default(self, obj: Any) -> Any:
        """handle array/Path type"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return json.JSONEncoder.default(self, obj)


def load_json(filepath: PathLike, **kwargs) -> dict[str, Any]:
    """
    Load a json as dict
    :param filepath: json filepath
    :param kwargs: additional arguments to ``json.load()``
    :return:
    """
    with open(filepath, "r") as file:
        print_load(filepath)
        return json.load(file, **kwargs)


def save_json(filepath: PathLike, dict_obj: dict[str, Any], **kwargs) -> None:
    """
    Save dict as a json file

    :param filepath: json filepath
    :param dict_obj: dictionary object
    :param kwargs: additional arguments to ``json.dump()``
    """
    with open(filepath, "w") as outfile:
        print_save(filepath)
        json.dump(dict_obj, outfile, sort_keys=True, indent=4, cls=JsonEncodeHandler, **kwargs)
