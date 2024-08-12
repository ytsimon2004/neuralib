import subprocess

from neuralib.segmentation.cellpose.core import AbstractCellPoseOption
from neuralib.util.cli_args import CliArgs
from neuralib.util.gpu import check_mps_available, check_nvidia_cuda_available
from neuralib.util.verbose import fprint

__all__ = ['CellPoseSubprocOption']


class CellPoseSubprocOption(AbstractCellPoseOption):
    DESCRIPTION = 'Run cellpose as a CLI subprocess. Detail refer to ``Cellpose.cli``'

    def run(self):
        self.eval()

    def eval(self):
        cmds = self.build_cli()
        code = subprocess.check_call(cmds)
        return code

    def build_cli(self) -> list[str]:
        ret = ['python', '-m', 'cellpose', '--verbose']

        #
        if self.file_mode:
            ret.extend(CliArgs('--image_path', str(self.file)).as_command())
        elif self.batch_mode:
            ret.extend(CliArgs('--dir', str(self.directory)).as_command())

        #
        ret.extend(CliArgs('--chan', str(self.chan_seg)).as_command())
        ret.extend(CliArgs('--chan2', str(self.chan_nuclear)).as_command())

        #
        ret.extend(CliArgs('--pretrained_model', self.model).as_command())

        #
        ret.extend(CliArgs('--diameter', str(self.diameter)).as_command())

        #
        if self.no_normalize:
            ret.extend(CliArgs('--no_norm').as_command())

        #
        if check_nvidia_cuda_available(backend='torch') or check_mps_available(backend='torch'):
            ret.extend(CliArgs('--use_gpu').as_command())

            if check_mps_available():
                ret.extend(CliArgs('--gpu_device', 'mps').as_command())

        fprint(ret)

        return ret


if __name__ == '__main__':
    CellPoseSubprocOption().main()
