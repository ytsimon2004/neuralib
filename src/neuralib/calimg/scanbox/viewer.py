from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import sbxreader

from neuralib.argp import int_tuple_type
from neuralib.calimg.scanbox.core import SBXInfo
from neuralib.util.util_type import PathLike


class SBXViewer:
    """
    wrapper for sbxreader.
    note to check meta & sbxinfo consistency
    """
    info: SBXInfo
    sbx_map: sbxreader.sbx_memmap
    """(F, P, C, W, H)"""

    def __init__(self, file: PathLike):
        if Path(file).suffix != '.sbx':
            raise ValueError('')

        self.info = SBXInfo.load(file.with_suffix('.mat'))
        self.sbx_map = sbxreader.sbx_memmap(file)

    @property
    def meta(self) -> dict[str, Any]:
        return self.sbx_map.metadata

    @property
    def version(self) -> int:
        return int(self.info.scanbox_version)

    @property
    def height(self) -> int:
        return self.info.sz[0]

    @property
    def width(self) -> int:
        return self.info.sz[1]

    @property
    def n_planes(self) -> int:
        if not len(self.info.otwave) and self.info.volscan == 0:
            return 1
        elif self.info.volscan == 1:
            return len(self.info.otwave)
        raise RuntimeError('')

    @property
    def n_channels(self) -> int:
        """sbx issue, a bit of hard-coded"""
        if self.info.nchan is None:
            if self.info.channels == 2:
                return 1
            elif self.info.channels == 1:
                return 2
            else:
                raise RuntimeError('')
        else:
            return self.info.nchan

    @property
    def n_frames(self) -> int:
        return int(self.info.config.frames / self.n_planes)

    def display(self, frames: slice | np.ndarray | None,
                plane: int,
                channel: int):
        # from rscvp.util.imglib.viewer import ImageSequencesViewer # TODO fix

        if frames is None:
            frames = np.arange(0, self.n_frames)

        data = self.sbx_map[frames, plane, channel, :, :]
        data = np.asarray(data)
        ImageSequencesViewer.load(data).main()

    def to_tiff(self, frames: slice | np.ndarray | None,
                plane: int,
                channel: int,
                output: PathLike):
        import tifffile

        if frames is None:
            frames = np.arange(0, self.n_frames)
        data = self.sbx_map[frames, plane, channel, :, :]
        tifffile.imwrite(output, data)


def main():
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument('-D', '--dir', type=Path, required=True, help='directory containing .sbx/.mat scanbox output',
                    dest='directory')
    ap.add_argument('-F', '--frames', metavar='SLICE', type=int_tuple_type, default=None,
                    help='image sequences slice type', dest='frames')
    ap.add_argument('-P', '--plane', type=int, required=True, help='which optic plane', dest='plane')
    ap.add_argument('-C', '--channel', type=int, required=True, help='which pmt channel', dest='channel')
    ap.add_argument('-O', '--output', default=None, help='tiff output, if None, display the sequence', dest='output')

    opt = ap.parse_args()

    sbx = SBXViewer(opt.directory)
    frames = np.arange(*opt.frames).astype(int) if opt.frames is not None else None

    if opt.output is not None:
        sbx.to_tiff(frames, opt.plane, opt.channel, opt.output)
    else:
        sbx.display(frames, opt.plane, opt.channel)


if __name__ == '__main__':
    main()
