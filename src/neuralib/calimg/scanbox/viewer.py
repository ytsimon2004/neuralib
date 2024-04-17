from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import sbxreader

from neuralib.argp import int_tuple_type
from neuralib.calimg.scanbox.core import SBXInfo
from neuralib.imglib.labeller import SequenceLabeller
from neuralib.util.util_type import PathLike
from neuralib.util.utils import uglob

__all__ = ['SBXViewer']


class SBXViewer:
    """wrapper for sbxreader

    `Dimension parameters`:

        F = number of frames

        P = number of optical planes

        C = number of PMT channels

        W = FOV width

        H = FOV height

    """

    info: SBXInfo
    """:class:`~neuralib.calimg.scanbox.core.SBXInfo`"""

    sbx_map: sbxreader.sbx_memmap
    """(F, P, C, W, H)"""

    def __init__(self, directory: PathLike):
        """
        :param directory: scanbox file (.sbx). In the same directory should contain the corresponding .mat file
        """

        if not Path(directory).is_dir():
            raise NotADirectoryError(f'{directory}')

        sbx = uglob(directory, '*.sbx')
        self.sbx_map = sbxreader.sbx_memmap(sbx)

        info = uglob(directory, '*.mat')
        self.info = SBXInfo.load(info)

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
        """number of frames/images"""
        return int(self.info.config.frames / self.n_planes)

    def play(self,
             frames: slice | np.ndarray | None,
             plane: int,
             channel: int):
        """
        Play the selected frames using customized CV2 player.

        See :class:`~neuralib.imglib.labeller.SequenceLabeller`

        :param frames: selected frames. If None, play all sequences
        :param plane: number of optical planes
        :param channel: number of PMT channel
        :return:
        """

        if frames is None:
            frames = np.arange(0, self.n_frames)

        data = self.sbx_map[frames, plane, channel, :, :]
        data = np.asarray(data)

        SequenceLabeller.load_sequences(data).main()

    def to_tiff(self,
                frames: slice | np.ndarray | None,
                plane: int,
                channel: int,
                output: PathLike):
        """
        Convert the selected frames to tiff file

        :param frames: selected frames. If None, convert all sequences
        :param plane: number of optical planes
        :param channel: number of PMT channel
        :param output: output filename
        :return:
        """
        import tifffile

        if frames is None:
            frames = np.arange(0, self.n_frames)
        data = self.sbx_map[frames, plane, channel, :, :]
        tifffile.imwrite(output, data)


def main():
    import argparse
    ap = argparse.ArgumentParser(description='view or save the sbx file, If specify the output using -O, save as tiff'
                                             'otherwise, play.')

    ap.add_argument('-D', '--dir', type=Path, required=True, help='directory containing .sbx/.mat scanbox output',
                    dest='directory')
    ap.add_argument('-F', '--frames', metavar='SLICE', type=int_tuple_type, default=None,
                    help='image sequences slice type', dest='frames')
    ap.add_argument('-P', '--plane', type=int, required=True, help='which optic plane', dest='plane')
    ap.add_argument('-C', '--channel', type=int, required=True, help='which pmt channel', dest='channel')
    ap.add_argument('-O', '--output', default=None, help='tiff output, if None, display the sequence', dest='output')

    opt = ap.parse_args()

    viewer = SBXViewer(opt.directory)
    frames = np.arange(*opt.frames).astype(int) if opt.frames is not None else None

    #
    if opt.output is None:
        viewer.play(frames, opt.plane, opt.channel)
    else:
        viewer.to_tiff(frames, opt.plane, opt.channel, opt.output)


if __name__ == '__main__':
    main()
