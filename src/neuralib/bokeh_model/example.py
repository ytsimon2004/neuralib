import random

from bokeh.model import Model
from bokeh.models import ColumnDataSource, GlyphRenderer
from bokeh.plotting import figure

from neuralib.bokeh_model import ViewComponent, View, BokehServer


class Graph(ViewComponent):
    data: ColumnDataSource
    render_dots: GlyphRenderer
    render_line: GlyphRenderer

    def __init__(self):
        self.x = []
        self.y = []
        self.data = ColumnDataSource(data=dict(x=[], y=[]))
        self.w = 0
        self.h = 0

    def plot(self, fig: figure, **kwargs):
        self.w = fig.width
        self.h = fig.height
        self.render_dots = fig.dot(
            x='x', y='y', source=self.data,
            size=self.w / 20)

        self.render_line = fig.line(
            x='x', y='y', source=self.data)

    def update(self):
        x = random.random() * self.w
        y = random.random() * self.h
        self.x.append(x)
        self.y.append(y)
        self.data.data = dict(x=self.x, y=self.y)


class Top(View):
    graph: Graph

    @property
    def title(self) -> str:
        return 'A'

    def setup(self) -> Model:
        fig = figure(
            width=800, height=800,
            x_range=(-10, 810),
            y_range=(-10, 810),
            toolbar_location='above')
        self.graph = Graph()
        self.graph.plot(fig)

        from bokeh.layouts import column
        return column(fig)

    def update(self):
        self.graph.update()
        self.run_timeout(1000, self.update)


if __name__ == '__main__':
    VIEW = Top()
    BokehServer().start(VIEW)
