import abc
from pathlib import Path
from typing import Literal, Union, Optional

import numpy as np
from bokeh.models import GlyphRenderer, ColumnDataSource
from bokeh.plotting import figure

from neuralib.bokeh_model.base import ViewComponent

__all__ = ['AbstractBrainView', 'BrainView', 'TiffBrainView', 'AtlasBrainView']


class AbstractBrainView(ViewComponent):
    data_brain_slice: ColumnDataSource
    render_brain_slice: GlyphRenderer

    def __init__(self):
        self.data_brain_slice = ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))
        self._width: Optional[float] = None
        self._height: Optional[float] = None

    @property
    def width(self) -> Optional[float]:
        return self._width

    @width.setter
    def width(self, value: float):
        self._width = value

    @property
    def height(self) -> Optional[float]:
        return self._height

    @height.setter
    def height(self, value: float):
        self._height = value

    def plot(self, fig: figure,
             palette='Greys256'):
        self.render_brain_slice = fig.image(
            'image', x='x', y='y', dw='dw', dh='dh',
            source=self.data_brain_slice,
            palette=palette, level="image",
        )

    @property
    @abc.abstractmethod
    def brain_image(self) -> np.ndarray:
        pass

    def update(self, x: float = 0, y: float = 0):
        brain = self.brain_image
        if brain is None:
            self.data_brain_slice.data = dict(
                image=[], dw=[], dh=[], x=[], y=[]
            )
            return

        h, w = brain.shape
        if self._width is None:
            self._width = w
        if self._height is None:
            self._height = h

        self.data_brain_slice.data = dict(
            image=[brain], dw=[self.width], dh=[self.height], x=[x], y=[y]
        )


class BrainView(AbstractBrainView):
    def __init__(self):
        super().__init__()
        self.reference: np.ndarray = None

    @property
    def brain_image(self) -> np.ndarray:
        return self.reference

    @brain_image.setter
    def brain_image(self, image: np.ndarray):
        self.reference = image


class TiffBrainView(AbstractBrainView):
    def __init__(self, file: Path):
        super().__init__()

        from PIL import Image

        self.file_path = file
        im = Image.open(file)
        im = im.convert("RGBA")
        self.reference = np.array(im, dtype=np.uint8)
        self._offset = 0

    @property
    def offset(self) -> int:
        return self._offset

    @offset.setter
    def offset(self, value: int):
        self._offset = value

    @property
    def brain_image(self) -> np.ndarray:
        return self.reference[self._offset]


class AtlasBrainView(AbstractBrainView):
    PLANE = Literal['coronal', 'sagittal', 'transverse']

    def __init__(self, source: str, check_latest=False, plane: PLANE = 'coronal'):
        super().__init__()

        from bg_atlasapi import BrainGlobeAtlas
        atlas = BrainGlobeAtlas(
            source,
            check_latest=check_latest,
        )

        self.resolution: int = atlas.resolution[0]
        assert self.resolution == atlas.resolution[1] == atlas.resolution[2]

        self.reference = atlas.reference

        self.grid_x: np.ndarray
        self.grid_y: np.ndarray
        self.plane = plane  # invoke setter
        self._offset = 0
        self._offset_h = 0
        self._offset_w = 0
        self._offset_m: Optional[np.ndarray] = None

    @property
    def n_ap(self) -> int:
        return self.reference.shape[0]

    @property
    def n_dv(self) -> int:
        return self.reference.shape[1]

    @property
    def n_ml(self) -> int:
        return self.reference.shape[2]

    @property
    def plane(self) -> PLANE:
        return self._plane

    @plane.setter
    def plane(self, value: PLANE):
        self._plane = value
        self.grid_y, self.grid_x = np.mgrid[0:self.height, 0:self.width]
        self._offset_w = 0
        self._offset_h = 0
        self._offset_m = None

    @property
    def n_frame(self) -> int:
        if self._plane == 'coronal':
            return self.n_ap
        elif self._plane == 'sagittal':
            return self.n_ml
        elif self._plane == 'transverse':
            return self.n_dv
        else:
            raise RuntimeError()

    @property
    def width_n(self) -> int:
        if self._plane == 'coronal':
            return self.n_ml
        elif self._plane == 'sagittal':
            return self.n_ap
        elif self._plane == 'transverse':
            return self.n_ml
        else:
            raise RuntimeError()

    @property
    def height_n(self) -> int:
        if self._plane == 'coronal':
            return self.n_dv
        elif self._plane == 'sagittal':
            return self.n_dv
        elif self._plane == 'transverse':
            return self.n_ap
        else:
            raise RuntimeError()

    @property
    def width(self) -> float:
        return self.width_n * self.resolution

    @property
    def height(self) -> float:
        return self.height_n * self.resolution

    @property
    def offset(self) -> int:
        return self._offset

    @offset.setter
    def offset(self, value: int):
        self._offset = value

    @property
    def offset_w(self) -> int:
        return self._offset_w

    @offset_w.setter
    def offset_w(self, value: int):
        self._offset_w = value
        self._offset_m = self.gen_offset_matrix(self._offset_w, self._offset_h)

    @property
    def offset_h(self) -> int:
        return self._offset_h

    @offset_h.setter
    def offset_h(self, value: int):
        self._offset_h = value
        self._offset_m = self.gen_offset_matrix(self._offset_w, self._offset_h)

    @property
    def offset_matrix(self) -> np.ndarray:
        if self._offset_m is None:
            self._offset_m = self.gen_offset_matrix(self._offset_w, self._offset_h)
        return self._offset + self._offset_m

    def gen_offset_matrix(self, h: int, v: int) -> np.ndarray:
        """

        :param h: horizontal plane diff to the center. right side positive.
        :param v: vertical plane diff to the center. bottom side positive.
        :return: (H, W) array
        """
        x_frame = np.round(np.linspace(-h, h, self.width_n)).astype(int)
        y_frame = np.round(np.linspace(-v, v, self.height_n)).astype(int)

        return x_frame[None, :] + y_frame[:, None]

    def brain_slice(self, offset: Union[int, np.ndarray]) -> np.ndarray:
        return self.reference[offset, self.grid_y, self.grid_x]

    @property
    def brain_image(self) -> np.ndarray:
        return self.brain_slice(self.offset_matrix)
