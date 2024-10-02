from pathlib import Path
from typing import ClassVar

import numpy as np
from cellpose import denoise

from neuralib.util.gpu import gpu_available
from .core import AbstractCellPoseOption, CellPoseEvalResult

__all__ = ['CellPoseAPIOption',
           'CellPoseEvalResult']


class CellPoseAPIOption(AbstractCellPoseOption):
    DESCRIPTION = 'run cell pose for directly call the api'

    RESTORE_TYPE: ClassVar[str] = 'denoise_cyto3'
    RESTORE_RATIO: ClassVar[float] = 1.0

    def run(self):
        if self.napari_view:
            self.launch_napari()
        elif self.cellpose_view:
            self.launch_cellpose_gui()
        else:
            self.eval()

    def eval(self):
        if self.file_mode:
            img = self.raw_image() if self.no_normalize else self.normalize_image()
            self._eval(self.file, img)

        elif self.batch_mode:
            iter_image = self.foreach_raw_image() if self.no_normalize else self.foreach_normalize_image()
            for name, img in iter_image:
                self._eval(name, img)

    def _eval(self, filepath: Path, image: np.ndarray):
        channel_choose = [self.chan_seg, self.chan_nuclear]
        model = denoise.CellposeDenoiseModel(
            gpu=True if gpu_available(backend='torch') else False,
            model_type=self.model,
            restore_type=self.RESTORE_TYPE,
            chan2_restore=True
        )

        masks, flows, styles, imgs_dn = model.eval(
            image,
            channels=channel_choose,
            diameter=self.diameter
        )

        ret = CellPoseEvalResult(
            filepath.name,
            image,
            self.diameter,
            channel_choose,
            masks,
            flows,
            styles,
            img_restore=imgs_dn,
            restore=self.RESTORE_TYPE,
            ratio=self.RESTORE_RATIO
        )

        ret.save_seg_file(str(filepath))

        if self.save_ij_roi:
            ret.save_roi(self.ij_roi_output(filepath))


if __name__ == '__main__':
    CellPoseAPIOption().main()
