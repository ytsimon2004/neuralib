import subprocess

from neuralib.segmentation.cellpose.core import AbstractCellPoseOption
from neuralib.util.gpu import check_mps_available, check_nvidia_cuda_available
from neuralib.util.verbose import fprint

__all__ = ['CellPoseSubprocOption']


class CellPoseSubprocOption(AbstractCellPoseOption):
    DESCRIPTION = 'Run cellpose as a CLI subprocess. Detail refer to ``Cellpose.cli``'

    def run(self):
        self.eval()

    def eval(self) -> int:
        cmds = self.build_cli()
        code = subprocess.check_call(cmds)
        return code

    def build_cli(self) -> list[str]:
        ret = ['python', '-m', 'cellpose', '--verbose']

        #
        if self.file_mode:
            ret.extend(['--image_path', str(self.file)])
        elif self.batch_mode:
            ret.extend(['--dir', str(self.directory)])

        #
        ret.extend(['--chan', str(self.chan_seg)])
        ret.extend(['--chan2', str(self.chan_nuclear)])

        #
        ret.extend(['--pretrained_model', self.model])

        #
        ret.extend(['--diameter', str(self.diameter)])

        #
        if self.no_normalize:
            ret.extend('--no_norm')

        #
        if check_nvidia_cuda_available(backend='torch') or check_mps_available(backend='torch'):
            ret.extend('--use_gpu')

            if check_mps_available():  # use mps in macOS
                ret.extend('--gpu_device', 'mps')

        fprint(ret)

        return ret


if __name__ == '__main__':
    CellPoseSubprocOption().main()
