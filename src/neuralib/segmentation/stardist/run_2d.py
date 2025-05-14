from pathlib import Path
from typing import TYPE_CHECKING

import napari
import numpy as np

from argclz import argument, as_argument
from neuralib.segmentation.base import AbstractSegmentationOptions
from neuralib.util.verbose import fprint
from .core import read_stardist, StarDistSegmentation, STARDIST_MODEL

if TYPE_CHECKING:
    from stardist.models import StarDist2D

__all__ = ['StarDist2DOptions']


class StarDist2DOptions(AbstractSegmentationOptions):
    DESCRIPTION = 'Run the Stardist model for segmentation'

    model: STARDIST_MODEL = as_argument(AbstractSegmentationOptions.model).with_options(
        default='2D_versatile_fluo',
        help='stardist pretrained model'
    )

    prob_thresh: float | None = argument(
        '--prob',
        default=None,
        help='Consider only object candidates from pixels with predicted object probability above this threshold. '
             'Seealso: stardist.models.base._predict_instances_generator: prob_thresh'
    )

    def run(self):
        if self.napari_view:
            self.launch_napari()
        else:
            self.eval()

    def seg_output(self, filepath: Path) -> Path:
        return filepath.with_name(filepath.stem + '_seg').with_suffix('.npz')

    def eval(self, **kwargs) -> None:
        from stardist.models import StarDist2D

        model = StarDist2D.from_pretrained(self.model)

        if self.file_mode:
            self._eval(self.file, self.process_image(), model)
        elif self.batch_mode:
            for file, image in self.foreach_process_image():
                self._eval(file, image, model)
        else:
            raise RuntimeError('run fail')

    def _eval(self, filepath: Path, image: np.ndarray, model: 'StarDist2D', **kwargs):
        out_seg = self.seg_output(filepath)

        if out_seg.exists() and not self.invalid_existed_result:
            fprint(f'cached {filepath} because {out_seg} exists, use --invalid to invalid cache', vtype='IO')
            return

        labels, detail = model.predict_instances(image, prob_thresh=self.prob_thresh, **kwargs)
        labels = labels.astype(np.float_)
        labels[labels == 0] = np.nan

        res = StarDistSegmentation(labels, detail['coord'], detail['prob'], str(filepath), self.model)

        # mask probability
        if self.prob_thresh is not None:
            res.mask_probability(self.prob_thresh)

        # save output
        res.to_npz(out_seg)

        if self.save_ij_roi:
            res.to_roi(self.ij_roi_output(filepath))

    # noinspection PyTypeChecker
    def launch_napari(self, with_widget: bool = False):
        """
        Launch napari viewer for stardist results

        :param with_widget: If True, launch also with the starDist widget (required package ``stardist-napari``)
        """
        file = self.seg_output(self.file)
        if not file.exists() or self.invalid_existed_result:
            self.eval()

        res = read_stardist(file)

        viewer = napari.Viewer()
        viewer.add_image(self.process_image(), name='image')
        viewer.add_image(res.labels, name='labels', colormap='twilight_shifted', opacity=0.5)
        viewer.add_points(res.points, face_color='red')

        if with_widget:
            viewer.window.add_plugin_dock_widget("stardist-napari", "StarDist")

        napari.run()


if __name__ == '__main__':
    StarDist2DOptions().main()
