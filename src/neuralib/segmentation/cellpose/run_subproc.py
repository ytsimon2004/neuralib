import subprocess
from pathlib import Path
from typing import Literal

import torch.cuda

from neuralib.argp import argument, as_argument
from neuralib.segmentation.base import AbstractSegmentationOption
from neuralib.segmentation.cellpose.core import AbstractCellPoseOption, CPOSE_MODEL
from neuralib.util.cli_args import CliArgs
from neuralib.util.gpu import check_mps_available
from neuralib.util.util_verbose import fprint

__all__ = ['CellPoseSubprocOption']


class CellPoseSubprocOption(AbstractCellPoseOption):
    DESCRIPTION = 'Run cellpose as a CLI subprocess. Detail refer to Cellpose.cli'

    file: Path = as_argument(AbstractSegmentationOption.file).with_options(
        metavar='FILE/DIR',
        required=True,
        help='run one single image if file or folder',
    )

    model: CPOSE_MODEL = as_argument(AbstractSegmentationOption.model).with_options(default='cyto3')

    label_type: Literal['gcamp6s', 'retro-aav'] = argument(
        '-T', '--type',
        default='retro-aav',
        help='type of fluorescence label type'
    )

    no_norm: bool = argument(
        '--no-norm',
        help='do not normalize images (normalize=False)'
    )

    def run(self):
        # single
        if self.file.is_file():

            if self.seg_result.exists():  # view
                if self.napari_view:
                    self.launch_napari()
                elif self.cellpose_gui:
                    self.launch_cellpose_gui()
                else:
                    raise RuntimeError(f'result for {self.file} existed!')
            else:
                self.eval()

        # batch
        else:
            self.eval()

    def eval(self):
        cmds = self.build_cli()
        code = subprocess.check_call(cmds)
        return code

    def build_cli(self) -> list[str]:
        ret = ['python', '-m', 'cellpose', '--verbose']

        #
        if self.file.is_dir():
            ret.extend(CliArgs('--dir', str(self.file)).as_command())
        elif self.file.is_file():
            ret.extend(CliArgs('--image_path', str(self.file)).as_command())

        #
        ret.extend(CliArgs('--chan', self.chan_seg).as_command())
        ret.extend(CliArgs('--chan2', self.chan_nuclear).as_command())

        #
        ret.extend(CliArgs('--pretrained_model', self.model).as_command())

        #
        ret.extend(CliArgs('--diameter', str(self.diameter)).as_command())

        #
        if self.no_norm:
            ret.extend(CliArgs('--no_norm').as_command())

        #
        if torch.cuda.is_available() or check_mps_available():
            ret.extend(CliArgs('--use_gpu').as_command())

            if check_mps_available():
                ret.extend(CliArgs('--gpu_device', 'mps').as_command())

        fprint(ret)

        return ret


if __name__ == '__main__':
    CellPoseSubprocOption().main()
