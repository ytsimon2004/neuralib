from functools import cached_property
from typing import Literal

import attrs
import napari
import numpy as np
from stardist.models import StarDist2D
from typing_extensions import Self

from neuralib.argp import argument, as_argument
from neuralib.segmentation.base import AbstractSegmentationOption

STARDIST_MODEL = Literal['2D_versatile_fluo', '2D_versatile_he', '2D_paper_dsb2018', '2D_demo']


@attrs.define
class StarDistResult:
    """
    Shape Info
    N: all cell numbers
    E: polygons edge numbers
    """
    labels: np.ndarray
    """image with label"""
    cord: np.ndarray
    """(N, 2, E) """
    prob: np.ndarray
    """(N,)"""

    point: np.ndarray = attrs.field(init=False)
    """coordinates to points by simple xy average"""

    def __attrs_post_init__(self):
        p = np.zeros((self.cord.shape[0], 2))
        for i, c in enumerate(self.cord):
            x_mean = np.mean(c[0, :])
            y_mean = np.mean(c[1, :])
            p[i, 0] = x_mean
            p[i, 1] = y_mean
        self.point = p

    def with_probability(self, threshold: float) -> Self:
        m = self.prob >= threshold
        return attrs.evolve(self, cord=self.cord[m], prob=self.prob[m])


class RunStarDist2DOptions(AbstractSegmentationOption):
    DESCRIPTION = 'Run the Stardist model for segmentation'

    model: STARDIST_MODEL = as_argument(AbstractSegmentationOption.model).with_options(
        default='2D_versatile_fluo',
        help='stardist pretrained model'
    )

    prob_thresh: float = argument('--prob', default=0.5,
                                  help='Consider only object candidates from pixels with predicted object probability '
                                       'above this threshold. '
                                       'Seealso: stardist.models.base._predict_instances_generator: prob_thresh')

    def run(self):
        if self.napari_view:
            self.launch_napari()

    @cached_property
    def raw_image(self) -> np.ndarray:
        return self.load_gray_scale(normalize=False)

    @property
    def normalized_images(self) -> np.ndarray:
        return self.load_gray_scale(normalize=True)

    def eval(self, prob: float = 0.5, **kwargs) -> StarDistResult:
        model = StarDist2D.from_pretrained(self.model)
        img = self.load_gray_scale()
        labels, detail = model.predict_instances(img, prob_thresh=self.prob_thresh, **kwargs)

        return StarDistResult(labels, detail['coord'], detail['prob']).with_probability(self.prob_thresh)

    def _remove_outlier(self):
        pass

    def launch_napari(self, with_widget=False):
        res = self.eval()
        viewer = napari.Viewer()
        viewer.add_image(self.raw_image, name='raw')
        viewer.add_image(self.normalized_images, name='normalized')
        viewer.add_image(res.labels, name='labels', colormap='cyan', opacity=0.5)
        viewer.add_points(res.point, face_color='red')

        if with_widget:
            viewer.window.add_plugin_dock_widget("stardist-napari", "StarDist")

        napari.run()


if __name__ == '__main__':
    RunStarDist2DOptions().main()
