import re
import sys
from pathlib import Path
from typing import Union, NamedTuple, Literal, overload, Final

import numpy as np
import polars as pl
from typing_extensions import Self

from neuralib.glx.base import EphysRecording
from neuralib.glx.channel_info import ChannelInfo

__all__ = ['GlxMeta', 'GlxRecording', 'GlxIndex', 'GlxFile']



class GlxMeta:
    """SpikeGLX meta file"""

    def __init__(self, meta: Union[str, Path, dict[str, str]]):
        if isinstance(meta, (str, Path)):
            meta = self._load_meta_dict(meta)

        self.__meta = meta

    @classmethod
    def _load_meta_dict(cls, path: Union[str, Path]) -> dict[str, str]:
        ret = {}
        with Path(path).open() as f:
            for line in f:
                k, _, v = line.rstrip().partition('=')
                ret[k] = v
        return ret

    @property
    def total_channels(self) -> int:
        return int(self.__meta['nSavedChans'])

    @property
    def total_samples(self) -> int:
        return int(self.__meta['fileSizeBytes']) // self.total_channels // 2

    @property
    def sample_rate(self) -> float:
        return int(self.__meta['imSampRate'])

    @property
    def total_duration(self) -> float:
        return float(self.__meta['fileTimeSecs'])

    @property
    def meta(self) -> dict[str, str]:
        return self.__meta

    def imro_table(self) -> str:
        return self.__meta['~imroTbl']

    if sys.version_info >= (3, 10):
        def channel_info(self) -> ChannelInfo:
            from neurocarto.probe_npx.npx import ChannelMap
            chmap = ChannelMap.from_imro(self.imro_table())
            return ChannelInfo(pl.DataFrame(dict(
                channel=np.arange(chmap.n_channels),
                shank=chmap.channel_shank,
                pos_x=chmap.channel_pos_x,
                pos_y=chmap.channel_pos_y,
            )))


class GlxRecording(GlxMeta, EphysRecording):
    """SpikeGLX binary file"""

    VOLTAGE_FACTOR: Final[float] = 0.195  # mV/1

    def __init__(self, path: Union[str, Path]):
        EphysRecording.__init__(self)

        path = Path(path)
        GlxMeta.__init__(self, path.with_suffix('.meta'))

        self.__path = path.with_suffix('.bin')
        self.__data = self._open_data()

    def _open_data(self) -> np.ndarray:
        path = self.__path
        n_channels = self.total_channels
        n_samples = self.total_samples
        return np.memmap(str(path), dtype='int16', mode='r', shape=(n_channels, n_samples), offset=0, order='F')

    @property
    def data_path(self) -> Path:
        return self.__path

    def __getitem__(self, item):
        return self.__data[item]


class GlxIndex(NamedTuple):
    """
    Represent an index from the "same" SpikeGLX recording session.

    Common naming pattern::

        <RUN_NAME>_g<G>_t<T>.imec<P>.(ap|lf).(bin|meta)

    For full file path pattern ::

        <RUN_NAME>_g<G>/<RUN_NAME>_g<G>_imec<P>/<RUN_NAME>_g<G>_t<T>.imec<P>.(ap|lf).(bin|meta)

    Example file name ::

        run_name_g0/run_name_g0_imec0/
        |_  run_name_g0_t0.imec0.ap.meta
        |_  run_name_g0_t0.imec0.ap.bin

    CatGT will change the ``t`` in the filename. For example::

        catgt_run_name_g0/
        |_run_name_g0_tcat.imec0.ap.meta
        |_run_name_g0_tcat.imec0.ap.bin
        |_run_name_g0_tcat.imec0.lf.meta
        |_run_name_g0_tcat.imec0.lf.bin

        supercatgt_run_name_g0/
        |_run_name_g0_tcat.imec0.ap.meta
        |_run_name_g0_tcat.imec0.ap.bin
        |_run_name_g0_tcat.imec0.lf.meta
        |_run_name_g0_tcat.imec0.lf.bin
    """

    run_name: str

    g: int = 0  # or str for internal use

    t: Literal['0', 'cat', 'super'] = '0'

    p: int = 0  # or str for internal use
    """probe index"""

    @classmethod
    def parse_filename(cls, name: str, use_supercat=False) -> Self:
        m = re.compile(r'(\w+)_g(\d+)_t(cat|super|\d+)\.imec(\d)\.(ap|lf)\.\w+').match(name)
        if m is None:
            raise RuntimeError(f'glx file name not follow the pattern : {name}')

        t = m.group(3)
        return GlxIndex(
            m.group(1),
            int(m.group(2)),
            ('super' if use_supercat else 'cat') if t == 'cat' else t,
            int(m.group(4)),
        )

    @property
    def is_catgt(self) -> bool:
        return self.t in ('cat', 'super')

    @property
    def is_supercat(self) -> bool:
        return self.t == 'super'

    def as_cat_index(self) -> Self:
        return self._replace(t='cat')

    def as_super_index(self) -> Self:
        return self._replace(t='super')

    def dirname(self, level: int = 0) -> str:
        """
        Build the glx directory name.

        **level**

        * 0: <RUN_NAME>_g<G>
        * 1: <RUN_NAME>_g<G>_imec<P>
        * other: <RUN_NAME>_g<G>/<RUN_NAME>_g<G>_imec<P>

        :param level: level of directory.
        :return:
        """
        name = f'{self.run_name}_g{self.g}'
        if self.t == 'cat':
            return f'catgt_{name}'
        elif self.t == 'super':
            return f'supercat_{name}'
        elif level == 0:
            return name
        elif level == 1:
            return f'{name}_imec{self.p}'
        else:
            return f'{name}/{name}_imec{self.p}'

    def filename(self, f: str = 'ap', ext='.bin') -> str:
        t = 'cat' if self.t == 'super' else self.t
        return f'{self.run_name}_g{self.g}_t{t}.imec{self.p}.{f}{ext}'

    @overload
    def replace(self, *, run_name=None, g=None, t=None, p=None) -> Self:
        pass

    def replace(self, **kwargs) -> Self:
        return self._replace(**kwargs)


class GlxFile(NamedTuple):
    """
    A SpikeGLX recording.
    """

    data_file: Path
    meta_file: Path
    glx_index: GlxIndex

    @classmethod
    def of(cls, file: Union[str, Path]) -> Self:
        file = Path(file)
        data_file = file.with_suffix('.bin')
        meta_file = file.with_suffix('.meta')
        glx_index = GlxIndex.parse_filename(data_file.name)
        return GlxFile(data_file, meta_file, glx_index)

    @property
    def root_directory(self) -> Path:
        r = self.run_name
        d = self.data_file.parent
        while d.name.startswith(r):
            d = d.parent
        return d

    @property
    def directory(self) -> Path:
        return self.data_file.parent

    @property
    def is_catgt_file(self) -> bool:
        """Is CatGT processed files?"""
        return self.glx_index.is_catgt

    @property
    def is_supercat_file(self) -> bool:
        """Is CatGT processed files?"""
        return self.glx_index.is_supercat

    @property
    def is_lfp_file(self) -> bool:
        return self.data_file.name.endswith('.lf.bin')

    def open(self) -> GlxRecording:
        return GlxRecording(self.data_file)

    def meta(self) -> GlxMeta:
        return GlxMeta(self.meta_file)

    @property
    def run_name(self) -> str:
        return self.glx_index.run_name

    @property
    def g_index(self) -> int:
        return self.glx_index.g

    @property
    def t_index(self) -> Literal['0', 'cat', 'super']:
        return self.glx_index.t

    @property
    def p_index(self) -> int:
        return self.glx_index.p

    def with_glx_index(self, glx_index: GlxIndex, root: Path = None) -> Self:
        if root is None:
            root = self.root_directory

        f = 'lf' if self.is_lfp_file else 'ap'
        data_file = root / glx_index.dirname(0) / glx_index.filename(f=f, ext='.bin')

        return self._replace(data_file=data_file, meta_file=data_file.with_suffix('.meta'), glx_index=glx_index)

    def as_cat_file(self) -> Self:
        if self.is_catgt_file:
            return self
        return self.with_glx_index(self.glx_index.as_cat_index())

    def as_supercat_file(self) -> Self:
        if self.is_supercat_file:
            return self
        return self.with_glx_index(self.glx_index.as_super_index())
