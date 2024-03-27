from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import TypeVar, Generic, TypedDict

import numpy as np
from scipy.io import loadmat

from neurolib.util.io import JsonEncodeHandler
from neurolib.util.util_type import PathLike

__all__ = [
    'SBXInfo',
    'sbx_to_json',
    #
    'SBXScreenShot'
]

from neurolib.util.util_verbose import fprint

# TODO ALSO CHECK the `res/rig/sbx/rig1/scanbox_config.m`

T = TypeVar('T')  # matstruct
A = TypeVar('A')  # sbx attr


@dataclass(frozen=True)
class SBXInfo(Generic[T, A]):
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

    def print_as_dict(self):
        from dataclasses import asdict
        pprint(asdict(self))

    @staticmethod
    def _try_find_attr(obj: T, attr: str) -> A | list:
        try:
            ret: A = getattr(obj, attr)
        except AttributeError:
            return []
        else:
            return ret

    # =============================== #
    # attributes from SBX .mat output #
    # =============================== #

    @classmethod
    def load(cls, root: Path | str) -> SBXInfo:
        info = loadmat(root, squeeze_me=True, struct_as_record=False)['info']

        try:
            nchan = info.chan.nchan  # version >= 3
        except AttributeError:
            nchan = None

        return SBXInfo(
            scanbox_version=str(info.scanbox_version),
            objective=info.objective,
            abort_bit=info.abort_bit,
            area_line=info.area_line,
            ballmotion=info.ballmotion,
            bytesPerBuffer=info.bytesPerBuffer,
            scanmode=info.scanmode,
            config=cls.get_config_info(info.config),
            calibration=cls.get_calibration_info_list(info.calibration),
            channels=info.channels,
            messages=info.messages,
            opto2pow=info.opto2pow,
            otparam=info.otparam,
            otwave=info.otwave,
            otwave_um=info.otwave_um,
            otwavestyle=info.otwavestyle,
            postTriggerSamples=info.postTriggerSamples,
            power_depth_link=info.power_depth_link,
            recordsPerBuffer=info.recordsPerBuffer,
            resfreq=info.resfreq,
            sz=info.sz,
            usernotes=info.usernotes,
            volscan=info.volscan,
            nchan=nchan
        )

    @dataclass(frozen=True)
    class CalibrationInfo:
        """attr from calibration"""
        x: float
        y: float
        gain_resonant_mult: int
        uv: list = field(default_factory=list)
        delta: list = field(default_factory=list)

    @classmethod
    def get_calibration_info_list(cls, calibration_list) -> list[CalibrationInfo]:
        return [cls.CalibrationInfo(
            x=calinfo.x,
            y=calinfo.y,
            gain_resonant_mult=calinfo.gain_resonant_mult,
            uv=calinfo.uv,
            delta=calinfo.delta
        ) for calinfo in calibration_list]

    # noinspection PyUnresolvedReferences
    @dataclass(frozen=True)
    class ConfigInfo:
        """attr from config"""
        agc: 'AgcInfo'
        # calibration: np.ndarray
        coord_abs: np.ndarray
        coord_rel: np.ndarray
        frame_times: np.ndarray
        frames: int  # total frames
        host_name: str  # BSTATION6
        knobby: 'KnobbyInfo'
        laser_power: float
        laser_power_perc: str  # 75%
        lines: int  # 528
        magnification: int  # idx from 1 in magnification_list
        magnification_list: np.ndarray
        objective: 'ObjectiveInfo'  # directly get useful attr `name`
        objective_type: int
        pmt0_gain: float  # green channel
        pmt1_gain: float  # red channel
        wavelength: int  # laser wavelength. i.e., 920 nm

    @classmethod
    def get_config_info(cls, config: T) -> ConfigInfo:
        return cls.ConfigInfo(
            agc=cls.get_agc_info(config.agc),
            # calibration=config.calibration,
            coord_abs=config.coord_abs,
            coord_rel=config.coord_rel,
            frame_times=config.frame_times,
            frames=config.frames,
            host_name=cls._try_find_attr(config, 'host_name'),
            knobby=cls.get_knobby_info(config.knobby),
            laser_power=config.laser_power,
            laser_power_perc=config.laser_power_perc,
            lines=config.lines,
            magnification=config.magnification,
            magnification_list=config.magnification_list,
            objective=cls.get_objective_info(config.objective),
            objective_type=config.objective_type,
            pmt0_gain=config.pmt0_gain,
            pmt1_gain=config.pmt1_gain,
            wavelength=config.wavelength,
        )

    @dataclass(frozen=True)
    class AgcInfo:
        """attr from info.config.agc"""
        agc_prctile: np.ndarray
        enable: int
        threshold: int

    @classmethod
    def get_agc_info(cls, agc: T) -> AgcInfo:
        return cls.AgcInfo(
            agc_prctile=agc.agc_prctile,
            enable=agc.enable,
            threshold=agc.threshold
        )

    # noinspection PyUnresolvedReferences
    @dataclass(frozen=True)
    class KnobbyInfo:
        """attr from info.config.knobby"""
        pos: 'KnobbyPosInfo'
        schedule: np.ndarray

    @classmethod
    def get_knobby_info(cls, knobby: T) -> KnobbyInfo:
        return cls.KnobbyInfo(
            pos=cls.get_knobby_pos_info(knobby.pos),
            schedule=knobby.schedule
        )

    @dataclass(frozen=True)
    class KnobbyPosInfo:
        """attr from info.config.knobby.pos
        manipulator coordinates
        """
        a: float
        x: float
        y: float
        z: float

    @classmethod
    def get_knobby_pos_info(cls, pos: T) -> KnobbyPosInfo:
        return cls.KnobbyPosInfo(
            a=pos.x,
            x=pos.x,
            y=pos.y,
            z=pos.z
        )

    @dataclass(frozen=True)
    class ObjectiveInfo:
        name: str

    @classmethod
    def get_objective_info(cls, objective: T) -> ObjectiveInfo:
        return cls.ObjectiveInfo(
            name=objective.name
        )

    # =============================== #
    # attributes from SBX .mat output #
    # =============================== #

    @property
    def fov_distance(self) -> tuple[float, float]:
        """(X, Y) in um"""
        if not self._validate_fov_distance():
            fprint('check the runconfig for ScanBox during recording', vtype='error')

        return 892, 667

    def _validate_fov_distance(self) -> bool:
        """due to this is the info only seen in GUI"""
        obj_type = self.objective == 'Nikon 16x/0.8w/WD3.0'
        n_line = self.get_config_info(self.config).lines == 528
        mag = self.get_config_info(self.config).magnification_list[self.magidx] == '1.7'
        return all([obj_type, n_line, mag])


def load_scanbox_config(root: Path) -> dict:
    """load for Rig-specific default setting config of the scanbox `scanbox_config.m`
    Note that the actual value could be changed via scanbox GUI"""
    ret = {}
    with root.open() as f:
        for line in f:
            line = line.strip().replace(' ', '')
            if line.startswith('sbconfig'):
                kidx = line.find('.')
                vidx = line.find('=')

                try:
                    eidx = line.index(';')
                    ret[line[kidx + 1:vidx]] = line[vidx + 1:eidx]
                except ValueError:
                    ret[line[kidx + 1:vidx]] = 'NOT IMPLEMENT'  # TODO matlab line change and `switch/case`
                    pass

    return ret


def sbx_to_json(matfile: Path | str,
                output: Path | None = None):
    """save .mat file to json"""
    if isinstance(matfile, str):
        matfile = Path(matfile)

    mat = SBXInfo.load(matfile)
    if output is None:
        output = matfile.with_name('sbx').with_suffix('.json')

    dy = dataclasses.asdict(mat)
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
