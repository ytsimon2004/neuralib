from pathlib import Path
from typing import Any

import numpy as np
import sbxreader
from rich.pretty import pprint

from argclz import AbstractParser, argument, pos_argument, validator, int_tuple_type
from neuralib.imaging.scanbox import SBXInfo
from neuralib.typing import PathLike

__all__ = ['ScanBoxView',
           'ScanBoxViewOptions']

from neuralib.util.utils import uglob


class ScanBoxView:
    """Scanbox info and data

    `Dimension parameters`:

        F = number of frames

        P = number of optical planes

        C = number of PMT channels

        W = FOV width

        H = FOV height
    """

    info: SBXInfo
    """:class:`~neuralib.imaging.scanbox.core.SBXInfo`"""

    sbx_map: sbxreader.sbx_memmap
    """`Array[float, [F, P, C, W, H]]`"""

    def __init__(self, directory: PathLike):
        """
        :param directory: directory contain ``.sbx`` & ``.mat`` files
        """
        if not Path(directory).is_dir():
            raise NotADirectoryError(f'{directory}')

        sbx = uglob(directory, '*.sbx')
        self.sbx_map = sbxreader.sbx_memmap(sbx)

        info = uglob(directory, '*.mat')
        self.info = SBXInfo.load(info)

    @property
    def meta(self) -> dict[str, Any]:
        """scanbox meta information"""
        return self.info.asdict()

    @property
    def version(self) -> int:
        """scanbox version"""
        return int(self.info.scanbox_version)

    @property
    def height(self) -> int:
        """fov height"""
        return self.info.sz[0]

    @property
    def width(self) -> int:
        """fov width"""
        return self.info.sz[1]

    @property
    def n_planes(self) -> int:
        """number of optical imaging planes"""
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
        """number of frames per plane"""
        return int(self.info.config.frames / self.n_planes)

    def show(self,
             frames: slice | np.ndarray | None,
             plane: int,
             channel: int):
        """play the selected frames using customized CV2 player

        :param frames: selected frames. If None, play all sequences
        :param plane: number of optical planes
        :param channel: number of PMT channel
        """
        from neuralib.imglib.labeller import SequenceLabeller

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
        """

        import tifffile

        if frames is None:
            frames = np.arange(0, self.n_frames)
        data = self.sbx_map[frames, plane, channel, :, :]
        tifffile.imwrite(output, data)


class ScanBoxViewOptions(AbstractParser):
    DESCRIPTION = 'ScanBox data view & save options'

    directory: Path = pos_argument(
        'PATH',
        validator.path.is_dir().is_exists(),
        help='directory containing .sbx/.mat scanbox output'
    )

    # -- Config ---------------
    frames: tuple[int, int] | None = argument(
        '--frames',
        type=int_tuple_type,
        default=None,
        help='indices of image sequences, if None, then all frames'
    )
    plane: int = argument('--plane', default=0, help='which optic plane')
    channel: int = argument('--channel', default=0, help='which pmt channel')

    # -- IO ---------------------
    verbose: bool = argument('--verbose', help='show meta verbose')
    show: bool = argument('--show', help='play the selected imaging sequences')
    to_tiff: Path | None = argument('--tiff', default=None, help='save sequence as tiff output')

    def run(self):
        viewer = ScanBoxView(self.directory)
        frames = np.arange(*self.frames).astype(int) if self.frames is not None else None

        if self.verbose:
            pprint(viewer.info.asdict())

        if self.show:
            viewer.show(frames, self.plane, self.channel)

        if self.to_tiff is not None:
            viewer.to_tiff(frames, self.plane, self.channel, output=self.to_tiff)


if __name__ == '__main__':
    ScanBoxViewOptions().main()
