"""
Get basic GPU info table and verbose
---------------

- cuda driver (Windows)
- metal backend support (MacOS)

.. code-block:: python

    from neuralib.util.gpu import setup_clogger
    print_gpu_table()


"""

from __future__ import annotations

import dataclasses
import platform
import subprocess
from typing import TypedDict

import GPUtil
import polars as pl
import torch

from neuralib.util.color_logging import setup_clogger
from neuralib.util.table import rich_data_frame_table

__all__ = [
    'print_gpu_table',
    'check_mps_available',
    'check_nvidia_cuda_available'
]


class GPUInfoWin(TypedDict, total=False):
    id: str
    name: str
    driver_version: str | None
    gpu_load: str
    """percentage of GPU usage"""
    total_memory: float
    """in MB"""
    free_memory: float
    used_memory: float
    temperature: float
    """in celsius"""


def _get_gpu_windows() -> list[GPUInfoWin]:
    ret = []
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        ret.append(
            GPUInfoWin(
                id=gpu.id,
                name=gpu.name,
                driver_version=gpu.driver,
                gpu_load=gpu.load,
                total_memory=gpu.memoryTotal,
                free_memory=gpu.memoryFree,
                used_memory=gpu.memoryUsed,
                temperature=gpu.temperature
            )
        )
    return ret


class GPUInfoMac(TypedDict, total=False):
    Chipset_Model: str
    Type: str
    Bus: str
    VRAM: str
    Vendor: str
    Device_ID: str
    Revision_ID: str
    Metal_Support: str

    mps_available: bool


def _get_gpu_mac() -> GPUInfoMac:
    """get mac gpu info from subprocess"""
    output = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], universal_newlines=True)
    lines = output.splitlines()

    ret = {}
    cur_gpu = None

    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        elif line.startswith('Graphics/Displays:'):
            cur_gpu = {}
        elif line.startswith('Displays:'):
            break
        elif cur_gpu is not None and line.strip():
            # print(line)
            key, value = line.split(':', 1)
            if value:
                ret[key.strip()] = value.strip()

    ret['mps_available'] = torch.backends.mps.is_available()

    return ret


def print_gpu_table() -> None:
    if platform.system() in ('win32', 'Windows'):
        info = _get_gpu_windows()
        check_nvidia_cuda_available()
    elif platform.system() == 'Darwin':
        info = _get_gpu_mac()
        check_mps_available()
    else:
        raise NotImplementedError('')

    #
    if isinstance(info, dict):
        rich_data_frame_table(info)
    elif isinstance(info, list):
        df = pl.concat([pl.DataFrame(dataclasses.asdict(it)) for it in info])
        rich_data_frame_table(df)
    else:
        raise TypeError('')


def check_mps_available() -> bool:
    logger = setup_clogger()
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            logger.error('MPS not available because pytorch install not built with MPS enable')
        else:
            logger.error('MPS not available because current MacOs version is not 12.3+,'
                         ' or do not have MPS-enabled device on this machine')
        return False
    else:
        logger.info('MPS is available')
        return True


def check_nvidia_cuda_available() -> bool:
    """
    Checks if the GPU driver reacts and otherwise raises a custom error.
    Useful to check before long GPU-dependent processes.
    """
    logger = setup_clogger()
    process = subprocess.Popen('nvidia-smi', shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, executable="/bin/bash")
    info, error = process.communicate()

    if process.returncode != 0:
        logger.error(f"{error.decode('utf-8')}")
        return False
    else:
        logger.info("nvidia-smi command successful")

    #
    if torch.cuda.is_available():
        logger.info('cuda is available in current env')
    else:
        logger.error('cuda computing in not setup properly')
