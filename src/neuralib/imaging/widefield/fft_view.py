from pathlib import Path

import cv2
import numpy as np
from bokeh.layouts import column
from bokeh.model import Model
from bokeh.models import ColumnDataSource, GlyphRenderer, Slider
from bokeh.plotting import figure
from tifffile import tifffile

from neuralib.argp import AbstractParser, argument
from neuralib.dashboard import ViewComponent, Figure, View, BokehServer
from neuralib.imaging.widefield import SequenceFFT
from neuralib.typing import PathLike

__all__ = ['WideFieldFFTViewOption']


class WideFieldFFTViewComponent(ViewComponent):
    source: ColumnDataSource
    render: GlyphRenderer

    def __init__(self, dat: np.ndarray | None):
        self.dat = dat
        self.source = ColumnDataSource(data=dict(x=[], y=[], map=[]))

    def plot(self, fig: Figure, **kwargs):
        self.render = fig.image_rgba(
            image='map',
            x=0,
            y=0,
            dw='x',
            dh='y',
            source=self.source,
            level='image'
        )

        fig.xgrid.grid_line_color = None
        fig.ygrid.grid_line_color = None
        fig.axis.visible = False

    def update(self, **kwargs):
        if (image := self.dat) is not None:
            self.update_image(image)

    def update_image(self, dat: np.ndarray):
        self.dat = dat
        if dat is not None:
            y, x = dat.shape
            self.source.data = dict(x=[x], y=[y], map=[dat])


class WideFieldFFTView(View):
    slider_saturation_factor: Slider
    slider_value_perc: Slider
    slider_saturation_perc: Slider

    fig_map: Figure
    view_map = WideFieldFFTViewComponent

    fft: SequenceFFT

    def __init__(self, seq_path: PathLike):
        self.seq_path = Path(seq_path)
        self.fft = SequenceFFT(tifffile.imread(seq_path))

    def setup(self) -> Model:
        w, h = self.fft.width, self.fft.height

        # slider
        self.slider_saturation_factor = Slider(
            start=0,
            end=1,
            value=0.3,
            step=0.1,
            width=600,
            title='saturation_factor'
        )
        self.slider_saturation_factor.on_change('value', self.on_select_saturation_factor)

        self.slider_value_perc = Slider(
            start=0,
            end=100,
            value=98,
            width=600,
            title='value_perc'
        )

        self.slider_value_perc.on_change('value', self.on_select_value_perc)

        self.slider_saturation_perc = Slider(
            start=0,
            end=100,
            value=90,
            width=600,
            title='saturation_perc'
        )

        self.slider_saturation_perc.on_change('value', self.on_select_saturation_perc)

        # fig
        self.fig_map = figure(title='image_intensity', width=w, height=h)
        self.fig_map = figure(
            title='image_map',
            width=w,
            height=h,
        )

        # view
        self.view_map = WideFieldFFTViewComponent(self.fft.seq)
        self.view_map.plot(self.fig_map)

        return column(
            self.slider_saturation_factor,
            self.slider_value_perc,
            self.slider_saturation_perc,
            self.fig_map
        )

    def on_select_saturation_factor(self, attr: str, old: str, value: str):
        self._on_select_slider_value(attr, old, value)

    def on_select_value_perc(self, attr: str, old: str, value: str):
        self._on_select_slider_value(attr, old, value)

    def on_select_saturation_perc(self, attr: str, old: str, value: str):
        self._on_select_slider_value(attr, old, value)

    def _on_select_slider_value(self, attr: str, old: str, value: str):
        cmap = self.as_colormap(self.slider_saturation_factor.value,
                                self.slider_value_perc.value,
                                self.slider_saturation_perc.value)
        self.run_later(self.view_map.update_image, cmap)

    def as_colormap(self, saturation_factor, value_perc, saturation_perc) -> np.ndarray:
        ret = self.fft.as_colormap(
            saturation_factor=saturation_factor,
            value_perc=value_perc,
            saturation_perc=saturation_perc
        )
        ret = cv2.cvtColor(ret, cv2.COLOR_RGB2RGBA)
        ret = ret.view(dtype=np.uint32).reshape((self.fft.height, self.fft.width))

        return np.flipud(ret)


class WideFieldFFTViewOption(AbstractParser):
    DESCRIPTION = 'View the HSV colormap representation of the Fourier transform results'

    file: str = argument(metavar='FILE', help='file path for the video sequence')

    def run(self):
        server = BokehServer(theme='caliber')
        server.start(WideFieldFFTView(self.file))


if __name__ == '__main__':
    WideFieldFFTViewOption().main()
