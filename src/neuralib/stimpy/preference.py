from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, Literal

from neuralib.util.util_type import PathLike

__all__ = [
    'MonitorDict',
    'MinotorWarpDict',
    'FlashIndicatorParameters',
    'NetworkControllerDict',
    'RigDict',
    'PreferenceDict',
    #
    'load_preferences'
]


class MonitorDict(TypedDict, total=False):
    sizePix: list[int]
    fullScreen: bool

    # PsychoPyDisplay
    name: str
    rate: float
    distance: int
    gamma: float
    width: int
    screen: int
    pos: list[int]
    winType: str

    # PyGameVRDisplay
    fov: float
    n_rays: float
    max_depth: float


class MinotorWarpDict(TypedDict, total=False):
    eyepoint: list[float]
    flipHorizontal: bool
    flipVertical: bool
    warp: str
    warpGridsize: int
    warpfile: str


class FlashIndicatorParameters(TypedDict, total=False):
    size: int
    units: str
    pos: list[int]
    fillColor: float
    mode: int
    state: bool
    frames: int


class NetworkControllerDict(TypedDict, total=False):
    ip: str
    port: int
    trigger: bool


class RigDict(TypedDict):
    port: str | Literal['dummy']


class PreferenceDict(TypedDict, total=False):
    user: str
    userPrefix: str
    expname: str  # git only
    defaultExperimentType: str
    default_imaging_mode: str

    logFolder: str | Path
    protocolsFolder: str | Path
    controllerFolder: str | Path
    stimsFolder: str | Path
    tmpFolder: str | Path  # git only

    monitor: list[MonitorDict]
    use_monitor: int | list[int]

    # PanoDisplay
    vr_flag: bool
    vrFolder: str | Path

    # PyGameVRDisplay
    textureFolder: str | Path
    raycaster: Literal['default', 'numpy', 'numba']

    # Photo indicator
    flashIndicator: bool
    flashIndicatorMode: int
    flashIndicatorParameters: FlashIndicatorParameters

    # Network
    labcams: NetworkControllerDict
    scanbox: NetworkControllerDict
    pycams: NetworkControllerDict
    spikeglx: NetworkControllerDict

    rig: RigDict
    warp: MinotorWarpDict

    # runtime append
    _controllers_data_folder: Path
    _controllers_protocol_folder: str
    _controllers_folder_read_flag: bool


def load_preferences(file: PathLike) -> PreferenceDict:
    """
    Get stimpy output(parsed) preference file as dict

    :param file: filepath for the .pref
    :return: :class:`PreferenceDict`
    """
    if Path(file).suffix != '.pref':
        raise ValueError('invalid file type')

    with open(file, "r") as file:
        return json.load(file)
