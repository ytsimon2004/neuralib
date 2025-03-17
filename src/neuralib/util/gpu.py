"""
Get basic GPU info table and verbose
--------------------------------------

- cuda driver (Windows/Linux)
- metal backend support (MacOS)

.. code-block:: python

    from neuralib.util.gpu import print_gpu_table
    print_gpu_table()


"""

import platform
import subprocess
from typing import TypedDict, Literal

import polars as pl
from neuralib.util.table import rich_data_frame_table
from neuralib.util.verbose import fprint

__all__ = [
    'print_gpu_table',
    'gpu_available',
    'check_mps_available',
    'check_nvidia_cuda_available'
]

RUN_BACKEND = Literal['tensorflow', 'torch']


def gpu_available(backend: RUN_BACKEND, *, check_smi: bool = False) -> bool:
    """

    :param backend: {'torch', 'tensorflow'}
    :param check_smi: check if ``nvidia-smi`` is runnable
    :return:
    """
    system = platform.system()
    if system in ('win32', 'Windows', 'Linux'):
        return check_nvidia_cuda_available(backend=backend, check_smi=check_smi)
    elif system == 'Darwin':
        return check_mps_available(backend=backend)


def print_gpu_table(backend: RUN_BACKEND, *, check_smi: bool = False) -> None:
    """
    Print GPU info table and check the compatibility with backend package

    :param backend: {'torch', 'tensorflow'}
    :param check_smi: check if ``nvidia-smi`` is runnable
    :return:
    """
    system = platform.system()
    if system in ('win32', 'Windows', 'Linux'):
        info = _get_gpu_windows()
        check_nvidia_cuda_available(backend=backend, check_smi=check_smi)
    elif system == 'Darwin':
        info = _get_gpu_mac(backend=backend)
    else:
        raise NotImplementedError(f'{system}!')

    #
    if isinstance(info, dict):
        rich_data_frame_table(info)
    elif isinstance(info, list):
        df = pl.concat([pl.DataFrame(it) for it in info])
        rich_data_frame_table(df)
    else:
        raise TypeError('')


# ============= #
# Windows/Linux #
# ============= #

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
    import GPUtil

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


def check_nvidia_cuda_available(backend: RUN_BACKEND,
                                check_smi: bool = False) -> bool:
    """
    Checks if the GPU driver reacts and otherwise raises a custom error.
    Useful to check before long GPU-dependent processes.

    :param backend: {'torch', 'tensorflow'}
    :param check_smi: check if ``nvidia-smi`` is runnable
    """

    if check_smi:
        process = subprocess.Popen('nvidia-smi',
                                   shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        info, error = process.communicate()

        if process.returncode != 0:
            fprint(f"{error.decode('utf-8')}", vtype='warning')
        else:
            fprint("nvidia-smi command successful", vtype='pass')

    #
    is_available = False

    if backend == 'torch':
        import torch
        if torch.cuda.is_available():
            is_available = True
    elif backend == 'tensorflow':
        import tensorflow as tf
        if tf.test.is_built_with_cuda():
            is_available = True
    else:
        raise ValueError(f'unknown backend: {backend}')

    #
    if is_available:
        fprint(f'cuda is available in current env using backend {backend}', vtype='pass')
        return True
    else:
        return False


# ====== #
# Mac OS #
# ====== #


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


def _get_gpu_mac(backend: RUN_BACKEND) -> GPUInfoMac:
    """get mac gpu info from subprocess

    :param backend: {'torch', 'tensorflow'}
    :return ``GPUInfoMac``
    """
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
            key, value = line.split(':', 1)
            if value:
                ret[key.strip()] = value.strip()

    ret['mps_available'] = check_mps_available(backend=backend)

    return ret


def check_mps_available(backend: RUN_BACKEND) -> bool:
    """
    Check if metal is available

    :param backend: {'torch', 'tensorflow'}
    :return: bool
    """
    is_available = True

    if backend == 'torch':
        import torch
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                fprint('MPS not available because pytorch install not built with MPS enable', vtype='warning')
            else:
                fprint('MPS not available because current MacOs version is not 12.3+,'
                       ' or do not have MPS-enabled device on this machine', vtype='warning')
            is_available = False

    elif backend == 'tensorflow':
        import tensorflow as tf
        if not tf.test.is_gpu_available():
            fprint('MPS not available in tensorflow backend', vtype='warning')
            is_available = False
    else:
        raise NotImplementedError(f'unknown backend: {backend}')

    #
    if is_available:
        fprint(f'MPS is available using backend: {backend}', vtype='pass')
        return True
    else:
        return False
