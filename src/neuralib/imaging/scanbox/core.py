from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import TypeVar, TypedDict

import numpy as np
from neuralib.io import JsonEncodeHandler
from neuralib.typing import PathLike
from scipy.io import loadmat

__all__ = [
    'SBXInfo',
    'sbx_to_json',
    #
    'screenshot_to_tiff'
]

T = TypeVar('T')  # dataclass


def copy_from(t: type[T], o: T, **specials) -> T:
    a = []
    for f in dataclasses.fields(t):
        if f.name in specials:
            a.append(specials[f.name])
        else:
            a.append(getattr(o, f.name))
    return t(*a)


@dataclass(frozen=True)
class SBXInfo:
    """for each recording session, `exp.mat` from scanbox"""
    scanbox_version: str
    objective: str
    abort_bit: int
    area_line: int
    ballmotion: np.ndarray
    bytesPerBuffer: int
    scanmode: int
    config: ConfigInfo
    calibration: list[CalibrationInfo]
    channels: int
    messages: np.ndarray
    opto2pow: np.ndarray
    otparam: np.ndarray  # ETL setting?
    otwave: np.ndarray
    otwave_um: np.ndarray
    otwavestyle: int
    postTriggerSamples: int
    power_depth_link: int
    recordsPerBuffer: int
    resfreq: int
    sz: np.ndarray  # line/frames?
    usernotes: np.ndarray
    volscan: int  # volumetric img? bool?

    # opt
    nchan: int | None

    @property
    def magidx(self) -> int:
        """info.config.magnification, used for `CalibrationInfo` idx"""
        return self.config.magnification - 1

    def print_asdict(self) -> None:
        from dataclasses import asdict
        pprint(asdict(self))

    # =============================== #
    # attributes from SBX .mat output #
    # =============================== #

    @dataclass(frozen=True)
    class CalibrationInfo:
        """attr from calibration"""
        x: float
        y: float
        gain_resonant_mult: int
        uv: list = field(default_factory=list)
        delta: list = field(default_factory=list)

    @dataclass(frozen=True)
    class AgcInfo:
        """attr from info.config.agc"""
        agc_prctile: np.ndarray
        enable: int
        threshold: int

    @dataclass(frozen=True)
    class KnobbyPosInfo:
        """attr from info.config.knobby.pos
        manipulator coordinates
        """
        a: float
        x: float
        y: float
        z: float

    # noinspection PyUnresolvedReferences
    @dataclass(frozen=True)
    class KnobbyInfo:
        """attr from info.config.knobby"""
        pos: KnobbyPosInfo  # noqa: F821
        schedule: np.ndarray

    @dataclass(frozen=True)
    class ObjectiveInfo:
        name: str

    # noinspection PyUnresolvedReferences
    @dataclass(frozen=True)
    class ConfigInfo:
        """attr from config"""
        agc: AgcInfo  # noqa: F821
        # calibration: np.ndarray
        coord_abs: np.ndarray
        coord_rel: np.ndarray
        frame_times: np.ndarray
        frames: int  # total frames
        host_name: str  # BSTATION6
        knobby: KnobbyInfo  # noqa: F821
        laser_power: float
        laser_power_perc: str  # 75%
        lines: int  # 528
        magnification: int  # idx from 1 in magnification_list
        magnification_list: np.ndarray
        objective: ObjectiveInfo  # directly get useful attr `name`  # noqa: F821
        objective_type: int
        pmt0_gain: float  # green channel
        pmt1_gain: float  # red channel
        wavelength: int  # laser wavelength. i.e., 920 nm

    @classmethod
    def load(cls, file: PathLike) -> SBXInfo:
        info = loadmat(file, squeeze_me=True, struct_as_record=False)['info']

        try:
            nchan = info.chan.nchan  # version >= 3
        except AttributeError:
            nchan = None

        return copy_from(SBXInfo, info,
                         scanbox_version=str(info.scanbox_version),
                         config=copy_from(cls.ConfigInfo, (config := info.config),
                                          agc=copy_from(cls.AgcInfo, config.agc),
                                          host_name=getattr(config, 'host_name', ''),
                                          knobby=copy_from(cls.KnobbyInfo, (knobby := config.knobby),
                                                           pos=copy_from(cls.KnobbyPosInfo, (pos := knobby.pos),
                                                                         a=pos.x)),
                                          objective=copy_from(cls.ObjectiveInfo, config.objective)),
                         calibration=[copy_from(cls.CalibrationInfo, cali) for cali in info.calibration],
                         nchan=nchan)

    # =============================== #
    # attributes from SBX .mat output #
    # =============================== #

    @property
    def fov_distance(self) -> tuple[float, float]:
        """(X, Y) in um.

        Note this value might be hardware dependent. value return is internal usage for the lab
        """
        lines = self.config.lines

        if self.objective == 'Nikon 16x/0.8w/WD3.0':
            obj = 16
        else:
            raise NotImplementedError('')

        zoom = float(self.config.magnification_list[self.magidx])

        return _get_default_scanbox_fov_dimension(lines, obj, zoom)

    def _validate_fov_distance(self) -> bool:
        """due to this is the info only seen in GUI"""
        return (
                self.objective == 'Nikon 16x/0.8w/WD3.0'
                and self.config.lines == 528
                and self.config.magnification_list[self.magidx] == '1.7'
        )


def _get_default_scanbox_fov_dimension(lines: int,
                                       obj_type: int,
                                       zoom: float) -> tuple[float, float]:
    """
    Hardware/settings dependent fov size according to recording configuration

    :param lines: number of lines for the scanning fov
    :param obj_type: objective magnification. i.e., 16X
    :param zoom: zoom setting during acquisition
    :return: (X, Y) in um
    """
    # ~ 30hz
    if lines == 528 and obj_type == 16:
        if zoom == 1:
            return 1396, 1056
        elif zoom == 1.2:
            return 1284, 978
        elif zoom == 1.4:
            return 1023, 765
        elif zoom == 1.7:
            return 892, 667
        elif zoom == 2.0:
            return 716, 531
        elif zoom == 2.4:
            return 632, 463

    raise NotImplementedError('check scanbox GUI directly')


def sbx_to_json(matfile: PathLike,
                output: Path | None = None,
                verbose: bool = True) -> None:
    """
    save .mat scanbox output file as json file

    :param matfile: .mat filepath
    :param output: output filepath, if None, create a json file in the same directory
    :param verbose: pprint as dict
    :return:
    """
    if isinstance(matfile, str):
        matfile = Path(matfile)

    mat = SBXInfo.load(matfile)
    if output is None:
        output = matfile.with_name('sbx').with_suffix('.json')

    dy = dataclasses.asdict(mat)

    if verbose:
        pprint(dy)

    with open(output, "w") as outfile:
        json.dump(dy, outfile, sort_keys=True, indent=4, cls=JsonEncodeHandler)


# ========== #
# ScreenShot #
# ========== #

class SBXScreenShot(TypedDict):
    __header__: str
    __version__: str
    __globals__: str
    img: np.ndarray


def screenshot_to_tiff(mat_file: PathLike,
                       output: PathLike | None = None) -> None:
    """
    Scanbox screenshot result (.mat) to tif file
    :param mat_file:
    :param output: save output path, otherwise show
    :return:
    """
    dat: SBXScreenShot = loadmat(mat_file)
    img = dat['img']

    if output is None:
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.show()
    else:
        import tifffile
        tifffile.imwrite(output, img)
