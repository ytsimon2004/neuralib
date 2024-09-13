"""

CatGT
=====

* `github <https://github.com/billkarsh/CatGT>`_
* `download <https://billkarsh.github.io/SpikeGLX/#catgt>`_

Download binary
---------------


1. download
2. Extra zip file into ``bin`` directory
3. go into ``CatGT-linux``
4. run ``bash install.sh``

Build from Source
-----------------

1. Clone Source Code

2. go into ``Build``, edit ``CatGT.pro``. change the ``DESTDIR`` ::


       unix {
           DESTDIR = .
       }

3. run ``qmake`` to generate Makefile ::

        make sure do not run it under conda environment.

4. run ``make`` to build binary.

   You can run ``make clean`` to clean objects.

5. copy ``CatGT`` to ``bin``

Usage CatGT in command-line
---------------------------

Input/Output
~~~~~~~~~~~~

It read ``GLX_DIR/GLX_NAME/GLX_NAME_t0.imecP.ap.bin`` and meta file.
It outputs file to ``GLX_DIR/catgt_GLX_NAME/GLX_NAME_tcat.imecP.ap.bin``.
Together with ap meta file, lf bin and lf meta file.

For SuperCat case, it read multiple cated file, and outputs files to
``GLX_DIR/supercat_GLX_NAME/GLX_NAME_tcat.imecP.ap.bin``.
Together with ap meta file, lf bin and lf meta file.

Default apply options
~~~~~~~~~~~~~~~~~~~~~

::

    bin/CatGT \
        -dest=GLX_DIR \
        -dir=GLX_DIR \
        -run=RUN_NAME \
        -g=0 -t=0 -ap -lf -prb=0 \
        -apfilter=butter,12,300,5000 \
        -lffilter=butter,12,1,1000 \
        -gblcar \
        -gfix=0.4,0.1,0.02 \
        -chnexcl={0;127}


SuperCat

::

    CatGT \
        -t=cat -no_run_fld -ap -lf -prb=0 \
        -supercat={GLX_DIR,RUN_NAME_g0}{...} \
        -dest=GLX_DIR

Use CatGT via CatGTOptions
--------------------------

put ``CatGT`` in the CATGT_PATH environment variable.


"""
import os
import subprocess
import time
from pathlib import Path
from typing import Final, Optional, Literal, Union

import numpy as np
from neuralib.util.util_verbose import fprint
from tqdm import tqdm

from neuralib.argp import argument, tuple_type, with_defaults
from .spikeglx import GlxFile, GlxMeta

__all__ = [
    'CatGTOptions', 'ensure_catgt_installed', 'parse_from_catgt_commands', 'wait_catgt_tqdm'
]

CATGT_PATH: Path = None


def ensure_catgt_installed() -> bool:  # TODO other way?
    global CATGT_PATH
    if CATGT_PATH is None:
        CATGT_PATH = Path(os.getenv('CATGT_PATH'))
    if not CATGT_PATH.is_file():
        raise RuntimeError(f'{CATGT_PATH} not a file')
    return True


class CatGTOptions:
    """

    **Example**

    .. code-block :: python

        catgt = CatGTOptions()
        catgt.set_source(glx_file)
        catgt.dest = output

        catgt.set_ap_filter(300, 5000)
        catgt.set_lf_filter(1, 1000)

        catgt.global_car = True
        catgt.set_gfix()

        subprocess.check_call(self.build_command())

    """
    GROUP_STREAM: Final[str] = 'Which streams'
    GROUP_OPTIONS: Final[str] = 'CatGT Options'
    GROUP_INPUT: Final[str] = 'CatGT Inputs'

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        with_defaults(obj)
        return obj

    def __init__(self, opts: Union[str, list[str], Path, GlxMeta] = None):
        if opts is not None:
            parse_from_catgt_commands(self, opts)

    ap: bool = argument(
        '-ap',
        group=GROUP_STREAM,
        help='required to process ap streams'
    )

    lf: bool = argument(
        '-lf',
        group=GROUP_STREAM,
        help='required to process lf streams'
    )

    prb_3A: bool = argument(
        '-prb_3A',
        group=GROUP_STREAM,
        help='if -ap or -lf process 3A-style probe files, e.g. run_name_g0_t0.imec.ap.bin'
    )

    prb: str = argument(
        '-prb',
        metavar='P,P:P,...',
        default='0',
        group=GROUP_STREAM,
        help='if -ap or -lf AND !prb_3A process these probes'
    )

    no_run_fld: bool = argument(
        '-no_run_fld',
        group=GROUP_OPTIONS,
        help='older data, or data files relocated without a run folder'
    )

    prb_fld: bool = argument(
        '-prb_fld',
        group=GROUP_OPTIONS,
        help='use folder-per-probe organization'
    )

    prb_miss_ok: bool = argument(
        '-prb_miss_ok',
        group=GROUP_OPTIONS,
        help="instead of stopping, silently skip missing probes"
    )

    g: str = argument(
        '-g',
        metavar='ga[,gb]',
        default='0',
        group=GROUP_OPTIONS,
        help="extract TTL from CatGT output files. from ga to gb (includes)"
    )

    t: str = argument(
        '-t',
        metavar='ta[,tb]|cat',
        default='0',
        group=GROUP_OPTIONS,
        help="extract TTL from CatGT output files. from ta to tb (includes)"
    )

    gt_list: str = argument(
        '-gtlist',
        metavar='{g,ta,tb}...',
        default=None,
        group=GROUP_OPTIONS,
        help="override {-g,-t} giving each listed g-index its own t-range"
    )

    exported: bool = argument(
        '-exported',
        group=GROUP_OPTIONS,
        help="apply FileViewer 'exported' tag to in/output filenames"
    )

    t_miss_ok: bool = argument(
        '-t_miss_ok',
        group=GROUP_OPTIONS,
        help="instead of stopping, zero-fill if trial missing"
    )

    zero_fill_max: int = argument(
        '-zerofillmax',
        group=GROUP_OPTIONS,
        help="set a maximum zero-fill span (millisec)"
    ).set_default(None, 500)

    max_secs: float = argument(
        '-maxsecs',
        # default=7.5,
        default=None,
        group=GROUP_OPTIONS,
        help="set a maximum output file length (float seconds)"
    )

    ap_filter: tuple[str, int, int, int] = argument(
        '-apfilter',
        metavar='TYPE,ORDER,F_HIGH,F_LOW',
        type=tuple_type((str, int, int, int)),
        default=None,
        group=GROUP_OPTIONS,
        help="apply ap band-pass filter of given {type, order, corners(float Hz)}"
    )

    lf_filter: tuple[str, int, int, int] = argument(
        '-lffilter',
        metavar='TYPE,ORDER,F_HIGH,F_LOW',
        type=tuple_type((str, int, int, int)),
        default=None,
        group=GROUP_OPTIONS,
        help="apply lf band-pass filter of given {type, order, corners(float Hz)}"
    )

    no_tshift: bool = argument(
        '-no_tshift',
        group=GROUP_OPTIONS,
        help="DO NOT time-align channels to account for ADC multiplexing"
    )

    local_car: tuple[int, int] = argument(
        '-loccar',
        group=GROUP_OPTIONS,
        type=tuple_type((int, int)),
        help="apply ap local CAR annulus (exclude radius, include radius). Deprecated"
    ).set_default(None, (2, 8))

    local_car_um: tuple[int, int] = argument(
        '-loccar_um',
        group=GROUP_OPTIONS,
        type=tuple_type((int, int)),
        help="apply ap local CAR annulus (exclude radius, include radius)"
    ).set_default(None, (40, 140))

    global_car: bool = argument(
        '-gblcar',
        group=GROUP_OPTIONS,
        help="apply ap global CAR filter over all channels"
    )

    gfix: tuple[float, float, float] = argument(
        '-gfix',
        type=tuple_type((float, float, float)),
        group=GROUP_OPTIONS,
        help="rmv ap artifacts: ||amp(mV)||, ||slope(mV/sample)||, ||noise(mV)||"
    ).set_default(None, (0.40, 0.10, 0.02))

    channel_exclude: str = argument(
        '-chnexcl',
        metavar='{prb;chans},...',
        default=None,
        group=GROUP_OPTIONS,
        help="this probe, exclude listed channels from ap local-car, global-car, gfix"
    )

    SY: list[str] = argument(
        '-SY',
        metavar='probe,word,bit,millisec[,tolerance]',
        # type=tuple_type((int, int, int, int)),
        # default=(0, 384, 6, 500),
        default=None,
        group=GROUP_OPTIONS,
        help="extract TTL signal from imec SY"
    )

    iSY: list[str] = argument(
        '-iSY',
        metavar='probe,word,bit,millisec[,tolerance]',
        # type=tuple_type((int, int, int, int)),
        # default=(0, 384, 6, 500),
        default=None,
        group=GROUP_OPTIONS,
        help="extract inverted TTL signal from imec SY"
    )

    inarow: int = argument(
        '-inarow',
        default=None,
        group=GROUP_OPTIONS,
        help="extractor antibounce stay high/low sample count"
    )

    pass1_force_ni_bin: bool = argument(
        '-pass1_force_ni_bin',
        group=GROUP_OPTIONS,
        help="write pass one ni binary tcat file even if not changed"
    )

    super_cat: str = argument(
        '-supercat',
        metavar='{dir,run_ga},...',
        default=None,
        group=GROUP_OPTIONS,
        help="concatenate existing output files across runs (see ReadMe)"
    )

    super_cat_trim_edges: bool = argument(
        '-supercat_trim_edges',
        group=GROUP_OPTIONS,
        help="supercat after trimming each stream to matched sync edges"
    )

    supercat_skip_ni_bin: bool = argument(
        '-supercat_skip_ni_bin',
        group=GROUP_OPTIONS,
        help="do not supercat ni binary files"
    )

    source_dir: Optional[Path] = argument(
        '-dir',
        metavar='PATH',
        group=GROUP_INPUT,
        help='Data Source Directory'
    )

    run_name: Optional[str] = argument(
        '-run',
        metavar='NAME',
        group=GROUP_INPUT,
    )

    dest: Path = argument(
        '-dest',
        metavar='PATH',
        group=GROUP_OPTIONS,
        help="alternate path for output files (must exist)"
    )

    out_probe_folder: bool = argument(
        '-out_prb_fld',
        group=GROUP_OPTIONS,
        help="if using -dest, create output sub-folder per probe"
    )

    save: list[str] = argument(
        '-save',
        metavar='JS,IP1,IP2,CH',
        action='append',
        group=GROUP_OPTIONS,
        help='save subset of probe chans'
    )

    def set_source(self, glx_file: GlxFile) -> GlxFile:
        """

        :param glx_file: glx source file
        :return: glx cat file
        """
        if glx_file.directory.name != glx_file.glx_index.dirname:
            raise RuntimeError(f'glx file does not put under {glx_file.glx_index.dirname} folder')

        self.source_dir = glx_file.directory.parent
        self.dest = glx_file.directory.parent
        self.run_name = glx_file.run_name
        self.g = str(glx_file.g_index)
        self.t = str(glx_file.t_index)
        self.prb = str(glx_file.p_index)

        return glx_file.as_cat_file()

    def set_supercat_source(self, glx_files: list[GlxFile]) -> GlxFile:
        """

        :param glx_files: glx cat files
        :return: glx supercat file
        """
        ret_file = build_supercat_file(glx_files)

        self.source_dir = None
        self.run_name = None
        self.prb = str(ret_file.p_index)
        self.no_run_fld = True
        self.super_cat = "".join([
            # {dir,run_ga}
            f'{{{it.directory},{it.run_name}_g{it.g_index}}}'
            for it in glx_files
        ])
        return ret_file

    def set_ap_filter(self, fo: int = 300, fh: int = 5000, filter_type: str = 'butter', order: int = 12):
        self.ap = True
        self.ap_filter = (filter_type, order, fo, fh)

    def set_lf_filter(self, fo: int = 1, fh: int = 1000, filter_type: str = 'butter', order: int = 12):
        self.lf = True
        self.lf_filter = (filter_type, order, fo, fh)

    def set_gfix(self, amp: float = 0.4, slope: float = 0.1, noise: float = 0.02):
        self.gfix = (amp, slope, noise)

    def set_channel_exclude(self, probe: int, channels: list[int], *, append=False):
        expr = f'{{{probe};{",".join(map(str, channels))}}}'
        if append and self.channel_exclude is not None:
            self.channel_exclude += f',{expr}'
        else:
            self.channel_exclude = expr

    def set_save_channels(self, stream: Literal['AP', 'LF'], probe: int, out_probe: int = None):
        """
        add `-save` option

        >>> catgt.set_save_channels('AP', 0)[0:100, 200:300, 384]
        # -save=2,0,0,0:100,200:300,384

        :param stream:
        :param probe:
        :param out_probe:
        :return:
        """
        return SetSaveChannelReceiver(self, stream, probe, out_probe)

    def build_command(self) -> list[str]:
        def comma_fmt(it):
            if it is None:
                return None
            return ','.join(map(str, it))

        def if_nn(obj, expr: str) -> Optional[str]:
            if obj:
                return expr.format(str(obj))
            else:
                return None

        args = [
            str(CATGT_PATH.absolute()),
            f'-dest={self.dest}',
            if_nn(self.source_dir, '-dir={}'),
            if_nn(self.run_name, '-run={}'),
            f'-g={self.g}',
            f'-t={self.t}',
            if_nn(self.gt_list, '-gtlist={}'),
            if_nn(self.out_probe_folder, '-out_prb_fld'),
            if_nn(self.ap, '-ap'),
            if_nn(self.lf, '-lf'),
            '-prb_3A' if self.prb_3A else f'-prb={self.prb}',
            if_nn(self.no_run_fld, '-no_run_fld'),
            if_nn(self.prb_fld, '-prb_fld'),
            if_nn(self.prb_miss_ok, '-prb_miss_ok'),
            if_nn(self.exported, '-exported'),
            if_nn(self.t_miss_ok, '-t_miss_ok'),
            if_nn(self.zero_fill_max, '-zerofillmax={}'),
            if_nn(self.max_secs, '-maxsecs={}'),
            if_nn(comma_fmt(self.ap_filter), '-apfilter={}'),
            if_nn(comma_fmt(self.lf_filter), '-lffilter={}'),
            if_nn(self.no_tshift, '-no_tshift'),
        ]

        if self.global_car:
            args.append('-gblcar')
        elif self.local_car_um is not None:
            args.append(if_nn(comma_fmt(self.local_car_um), '-loccar_um={}'))
        elif self.local_car is not None:
            args.append(if_nn(comma_fmt(self.local_car), '-loccar={}'))

        args.extend(self.save)

        if self.SY:
            args.append(if_nn(self.SY, '-SY={}'))
        elif self.iSY:
            args.append(if_nn(self.iSY, '-iSY={}'))

        args.extend([
            if_nn(self.gfix, '-gfix={}', formatter=comma_fmt),
            if_nn(self.channel_exclude, '-chnexcl={}'),
            if_nn(self.inarow, '-inarow={}'),
            if_nn(self.pass1_force_ni_bin, '-pass1_force_ni_bin'),
            if_nn(self.super_cat, '-supercat={}'),
            if_nn(self.super_cat_trim_edges, '-super_cat_trim_edges'),
            if_nn(self.supercat_skip_ni_bin, '-supercat_skip_ni_bin'),
        ])

        return list(filter(lambda it: it is not None, args))


def build_supercat_file(glx_files: list[GlxFile]) -> GlxFile:
    run_name = set()
    imro_name = set()
    g_index = set()
    p_index = set()

    for glx_file in glx_files:
        if not glx_file.is_catgt_file:
            raise RuntimeError(f'not a CatGTed file : {glx_file.data_file}')
        if glx_file.is_supercat_file:
            raise RuntimeError(f'is a supercat file : {glx_file.data_file}')

        run_name.add(glx_file.run_name)
        imro_name.add(glx_file.meta().imro_table())
        g_index.add(glx_file.g_index)
        p_index.add(glx_file.p_index)

    if len(run_name) > 1:
        raise RuntimeError(f'multiple run name : {list(run_name)}')
    if len(imro_name) > 1:
        raise RuntimeError(f'multiple imro used : {list(imro_name)}')
    if len(p_index) > 1:
        raise RuntimeError(f'multiple probe index : {list(p_index)}')

    d = glx_files[0].directory.parent
    i = glx_files[0].glx_index.as_super_index()
    bin_file = d / i.dirname / i.filename()

    return GlxFile(bin_file, bin_file.with_suffix('.meta'), i, )


def parse_from_catgt_commands(ret: CatGTOptions, line: Union[str, list[str], Path, GlxMeta]) -> CatGTOptions:
    if isinstance(line, Path):
        line = GlxMeta(line)

    # catGTCmdline0=<CatGT -dir=root/source.remote02/TS34/1_channelmap/20230118 -run=TS34_20230118_0 -g=0 -t=0 -ap -lf -prb=0 -apfilter=butter,12,300,5000 -lffilter=butter,12,1,1000 -gblcar -gfix=0.4,0.1,0.02 -chnexcl={0;127,141} -dest=root/source.remote02/_preprocess/TS34/1_channelmap/20230118>
    if isinstance(line, GlxMeta):
        line = line.meta['catGTCmdline0']
        if line.startswith('<') and line.endswith('>'):
            line = line[1: -1]

    if isinstance(line, str):
        line = line.split(' ')

    for item in line:
        if item == 'CatGT':
            continue
        key, _, value = item.partition('=')
        if key == '-dir':
            ret.source_dir = Path(value)
        elif key == '-run':
            ret.run_name = value
        elif key == '-g':
            ret.g = value
        elif key == '-t':
            ret.t = value
        elif key == '-ap':
            ret.ap = True
        elif key == '-lf':
            ret.lf = True
        elif key == '-prb':
            ret.prb = value
        elif key == '-apfilter':
            value = value.split(',')
            ret.ap_filter = (value[0], int(value[1]), int(value[2]), int(value[3]))
        elif key == '-lffilter':
            value = value.split(',')
            ret.lf_filter = (value[0], int(value[1]), int(value[2]), int(value[3]))
        elif key == '-gblcar':
            ret.global_car = True
        elif key == '-gfix':
            value = value.split(',')
            ret.gfix = (float(value[0]), float(value[1]), float(value[2]))
        elif key == '-chnexcl':
            ret.channel_exclude = value
        elif key == '-dest':
            ret.dest = Path(value)
        else:
            fprint(f'unrecognised CatGT Options : {item}')

    return ret


class SetSaveChannelReceiver:
    def __init__(self, opt: CatGTOptions, stream: Literal['AP', 'LF'], in_probe: int, out_probe: int = None):
        if out_probe is not None and out_probe < 0:
            raise ValueError()

        self.opt: Final = opt
        self.stream: Final = stream
        self.js: Final = {'AP': 2, 'LF': 3}[stream]
        self.in_probe: Final = in_probe
        self.out_probe: Final = out_probe if out_probe is not None else in_probe

        self._ch_list = None
        self._ch_expr = None

    def __contains__(self, item):
        return self._ch_list is not None and item in self._ch_list

    def __len__(self):
        return 0 if self._ch_list is None else len(self._ch_list)

    def __iter__(self):
        return iter(ret if (ret := self._ch_list) is not None else [])

    def __getitem__(self, item):
        if self._ch_list is not None:
            raise RuntimeError()

        if isinstance(item, np.ndarray):
            item = tuple(item)
        elif not isinstance(item, tuple):
            item = (item,)

        if self.opt.save is None:
            self.opt.save = []

        ch_list = []
        ch_expr = []

        for it in item:
            if isinstance(it, int) or np.isscalar(it):
                ch_list.append(int(it))
                ch_expr.append(str(ch_list[-1]))
            elif isinstance(it, slice):
                ch_expr.append(f'{it.start}:{it.stop}')
                ch_list.extend(list(range(it.start, it.stop + 1)))
            else:
                raise TypeError(repr(it))

        self._ch_list = ch_list
        self._ch_expr = ','.join(ch_expr)
        self.opt.save.append(f'{self.js},{self.in_probe},{self.out_probe},{self._ch_expr}')
        return self


UNIT = Literal['S', 'KS', 'MS', 'sec', 'min']


def wait_catgt_tqdm(opts: CatGTOptions, proc: subprocess.Popen, glx_file: GlxFile = None, cat_file: GlxFile = None, *,
                    unit: UNIT = 'sec'):
    """
    Monitor CatGt process with tqdm progress bar.

    **Example**

    .. code-block:: python

        catgt: CatGTOptions
        cat_file = catgt.set_source(glx_file)
        p = subprocess.run(self.build_command())

        wait_catgt_tqdm(catgt, p, glx_file, cat_file)

    :param opts:
    :param proc:
    :param glx_file:
    :param cat_file:
    :param unit:
    :return:
    """
    prog = CatgtProgress(glx_file, cat_file, 'ap', unit=unit)

    with tqdm(total=prog.total, desc='catgt-AP', unit=prog.unit) as t:
        while proc.poll() is None:
            t.update(prog.update())
            if prog.is_done():
                break
            time.sleep(1)

    if not opts.lf:
        return proc.wait()

    if (ret := proc.poll()) is not None:
        return ret

    prog = CatgtProgress(glx_file, cat_file, 'lf', unit=unit)

    with tqdm(total=prog.total, desc='catgt-LF', unit=prog.unit) as t:
        while proc.poll() is None:
            t.update(prog.update())
            if prog.is_done():
                break
            time.sleep(1)


class CatgtProgress:
    def __init__(self, glx_file: GlxFile, cat_file: GlxFile,
                 ap: Literal['ap', 'lf'] = 'ap',
                 unit: UNIT = 'sec'):
        if ap not in ('ap', 'lf'):
            raise ValueError()

        src_path = glx_file.data_file
        meta_path = glx_file.meta_file

        if not src_path.exists():
            raise FileNotFoundError()

        dst_path = cat_file.directory / cat_file.glx_index.filename(f=ap)

        src_meta = GlxMeta(meta_path)
        self.n_channels = src_meta.total_channels

        src_size = src_path.stat().st_size
        src_sample = int(src_size / self.n_channels / 2)
        src_rate = src_meta.sample_rate
        dst_rate = src_rate

        if ap == 'lf':
            dst_rate = 2500

        self._unit = unit
        self._src_sample = src_sample
        self._dst_rate = dst_rate
        self._dst_path = dst_path
        self._dst_sample = 0

        total = src_sample
        if unit == 'KS':
            total = int(src_sample * dst_rate / src_rate / 1000)
        elif unit == 'MS':
            total = int(src_sample * dst_rate / src_rate / 1000_000)
        elif unit == 'sec':
            total = int(src_sample / src_rate)
        elif unit == 'min':
            total = int(src_sample / src_rate / 60)

        self.total = total

    @property
    def unit(self) -> str:
        if self._unit == 'S':
            return 'samples'
        elif self._unit == 'KS':
            return 'K-samples'
        elif self._unit == 'MS':
            return 'M-samples'
        elif self._unit == 'sec':
            return 'recording seconds'
        elif self._unit == 'min':
            return 'recording minutes'

        return self._unit

    def current(self) -> int:
        if not self._dst_path.exists():
            return 0

        dst_size = self._dst_path.stat().st_size
        dst_sample = int(dst_size / self.n_channels / 2)

        if self._unit == 'KS':
            dst_sample //= 1000
        elif self._unit == 'MS':
            dst_sample //= 1000_000
        elif self._unit == 'sec':
            dst_sample = int(dst_sample / self._dst_rate)
        elif self._unit == 'min':
            dst_sample = int(dst_sample / self._dst_rate / 60)

        self._dst_sample = dst_sample
        return dst_sample

    def update(self) -> int:
        old = self._dst_sample
        return self.current() - old

    def is_done(self) -> bool:
        return self._dst_sample >= self.total
