from pathlib import Path
from typing import Literal, final

import attrs
import napari
import numpy as np
from numpy.lib.npyio import NpzFile
from stardist.models import StarDist2D
from typing_extensions import Self

from argclz import argument, as_argument
from neuralib.segmentation.base import AbstractSegmentationOptions
from neuralib.typing import PathLike
from neuralib.util.logging import setup_clogger, LOGGING_IO_LEVEL

Logger = setup_clogger(caller_name=__name__)
STARDIST_MODEL = Literal['2D_versatile_fluo', '2D_versatile_he', '2D_paper_dsb2018', '2D_demo']

__all__ = [
    'StarDist2DOptions',
    'StarDistResult'
]


class StarDist2DOptions(AbstractSegmentationOptions):
    DESCRIPTION = 'Run the Stardist model for segmentation'

    model: STARDIST_MODEL = as_argument(AbstractSegmentationOptions.model).with_options(
        default='2D_versatile_fluo',
        help='stardist pretrained model'
    )

    prob_thresh: float = argument(
        '--prob',
        default=0.8,
        help='Consider only object candidates from pixels with predicted object probability above this threshold. '
             'Seealso: stardist.models.base._predict_instances_generator: prob_thresh'
    )

    def run(self):
        if self.napari_view:
            self.launch_napari()
        else:
            self.eval()

    def seg_output(self, filepath: Path) -> Path:
        return (
            filepath
            .with_name(filepath.stem + f'_{self.model}')
            .with_suffix('.npz')
        )

    def eval(self, **kwargs) -> None:
        model = StarDist2D.from_pretrained(self.model)

        if self.file_mode:
            self._eval(self.file, self.process_image(), model)
        elif self.batch_mode:
            for file, image in self.foreach_process_image():
                self._eval(file, image, model)
        else:
            raise RuntimeError('run fail')

    def _eval(self, filepath: Path, image: np.ndarray, model: StarDist2D, **kwargs):
        out_seg = self.seg_output(filepath)

        if out_seg.exists() and not self.invalid_existed_result:
            Logger.info(f'Skip {filepath} because {out_seg} exists')
            return

        labels, detail = model.predict_instances(image, prob_thresh=self.prob_thresh, **kwargs)
        labels = labels.astype(np.float_)
        labels[labels == 0] = np.nan

        res = StarDistResult(
            filepath.name,
            labels,
            detail['coord'],
            detail['prob']
        ).with_probability(self.prob_thresh)

        res.savez(filepath.with_name(filepath.stem + f'_{self.model}'))

        if self.save_ij_roi:
            out_ij = self.ij_roi_output(filepath)
            res.save_roi(out_ij)

    # noinspection PyTypeChecker
    def launch_napari(self, with_widget: bool = False):
        """
        Launch napari viewer for stardist results

        :param with_widget: If True, launch also with the starDist widget (required package ``stardist-napari``)
        """
        file = self.seg_output(self.file)
        if not file.exists() or self.invalid_existed_result:
            self.eval()

        res = StarDistResult.load(file)

        viewer = napari.Viewer()
        viewer.add_image(self.process_image(), name='image')
        viewer.add_image(res.labels, name='labels', colormap='twilight_shifted', opacity=0.5)
        viewer.add_points(res.points, face_color='red')

        if with_widget:
            viewer.window.add_plugin_dock_widget("stardist-napari", "StarDist")

        Logger.info('Launch napari!')
        napari.run()


@final
@attrs.define
class StarDistResult:
    """
    Stardist results

    `Dimension parameters`:

        N = Number of segmented cell

        E = Number of polygons edge

        W = Image width

        H = Image height

        P = Number of image pixel with label

    """
    filename: str
    """Source image file name"""

    labels: np.ndarray
    """Image with label. `Array[float, [H, W]]`"""

    cords: np.ndarray
    """Coordinates. `Array[float, [N, 2, E]]`"""

    prob: np.ndarray
    """Detected probability. `Array[float, N]`"""

    points: np.ndarray = attrs.field(init=False)
    """Coordinates to points by simple XY average. `Array[float, [N, 2]]`"""

    def __attrs_post_init__(self):
        p = np.zeros((self.cords.shape[0], 2))
        for i, c in enumerate(self.cords):
            x_mean = np.mean(c[0, :])
            y_mean = np.mean(c[1, :])
            p[i, 0] = x_mean
            p[i, 1] = y_mean
        self.points = p

    @classmethod
    def load(cls, file: PathLike) -> Self:
        """
        load stardist results from npz file.
        :param file: ``.npz`` file
        :return: :class:`StarDistResult`
        """
        dat = np.load(file)
        Logger.log(LOGGING_IO_LEVEL, f'Load stardist results from {file}')
        return cls(
            filename=dat['filename'],
            labels=cls._reconstruct_labels_from_index_value(dat),
            cords=dat['cords'],
            prob=dat['prob'],
        )

    @classmethod
    def _reconstruct_labels_from_index_value(cls, dat: NpzFile) -> np.ndarray:
        h, w = dat['shape']
        index = dat['index']
        value = dat['value']

        image = np.full((h, w), np.nan)
        for i, (hi, wi) in enumerate(index):
            image[hi, wi] = value[i]

        return image

    def savez(self, filepath: Path) -> None:
        """
        Save ``filename``, ``cord``, ``prob``, ``point``, ``shape``, ``index``, ``index``, ``value`` as a npz file.

        shape: `Array[int, 2] in H,W`

        index: index with labels. `Array[int, [P, 2]]`

        value: label value in its index `Array[int, P]`

        """
        shape = np.array(self.labels.shape)
        index, value = self.get_index_value()
        output = filepath.with_suffix('.npz')

        np.savez(output,
                 filename=self.filename,
                 cords=self.cords,
                 prob=self.prob,
                 points=self.points,
                 shape=shape,
                 index=index,
                 value=value)

        Logger.log(LOGGING_IO_LEVEL, f'save stardist results to {output}')

    def save_roi(self, output_file: PathLike) -> None:
        """Save as imageJ ``.roi`` file"""
        from roifile import ImagejRoi, ROI_TYPE, ROI_OPTIONS

        points = np.fliplr(self.points)  # XY rotate in .roi format
        roi = ImagejRoi(
            roitype=ROI_TYPE.POINT,
            options=ROI_OPTIONS.PROMPT_BEFORE_DELETING | ROI_OPTIONS.SUB_PIXEL_RESOLUTION,
            n_coordinates=self.points.shape[0],
            integer_coordinates=points,
            subpixel_coordinates=points
        )

        roi.tofile(output_file)
        Logger.log(LOGGING_IO_LEVEL, f'save roi results to {output_file}')

    def get_index_value(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get labelled pixel index and its value.

        :return: `Array[int, [P, 2]]` and `Array[float, P]`
        """
        labels = self.labels
        h, w = labels.shape
        index = []
        value = []
        for i in range(h):
            for j in range(w):
                if not np.isnan(labels[i, j]):
                    index.append([i, j])
                    value.append(labels[i, j])

        return np.array(index).astype(np.int_), np.array(value).astype(np.int_)

    def with_probability(self, threshold: float) -> Self:
        """
        mask with threshold probability

        :param threshold: threshold probability for stardist result
        :return:
        """
        m = self.prob >= threshold
        return attrs.evolve(self, cords=self.cords[m], prob=self.prob[m])


if __name__ == '__main__':
    StarDist2DOptions().main()
