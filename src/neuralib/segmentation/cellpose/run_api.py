from typing import ClassVar

from cellpose import denoise

from .core import AbstractCellPoseOption, CellPoseEvalResult

__all__ = ['CellPoseEvalResult']


class CellPoseAPIOption(AbstractCellPoseOption):
    DESCRIPTION = 'run cell pose for directly call the api'

    RESTORE_TYPE: ClassVar[str] = 'denoise_cyto3'
    RESTORE_RATIO: ClassVar[float] = 1.0

    def run(self):
        if self.seg_result.exists():
            if self.napari_view:
                self.launch_napari()
            elif self.cellpose_gui:
                self.launch_cellpose_gui()
        else:
            res = self.eval()
            res.save_seg_file()

    def get_model(self) -> denoise.CellposeDenoiseModel:
        return denoise.CellposeDenoiseModel(gpu=True,
                                            model_type=self.model,
                                            restore_type=self.RESTORE_TYPE,
                                            chan2_restore=True)

    def eval(self) -> CellPoseEvalResult:
        img = self.load_gray_scale()
        channel_choose = [int(self.chan_seg), int(self.chan_nuclear)]
        model = self.get_model()
        masks, flows, styles, imgs_dn = model.eval(img,
                                                   channels=channel_choose,
                                                   diameter=self.diameter)
        return CellPoseEvalResult(
            str(self.file.with_suffix('')),
            img,
            self.diameter,
            channel_choose,
            masks,
            flows,
            styles,
            img_restore=imgs_dn,
            restore_type=self.RESTORE_TYPE,
            ratio=self.RESTORE_RATIO
        )


if __name__ == '__main__':
    CellPoseAPIOption().main()
