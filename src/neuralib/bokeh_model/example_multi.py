from bokeh.model import Model
from bokeh.models import Div

from neuralib.bokeh_model import View, BokehServer


class Top(View):

    @property
    def title(self) -> str:
        return 'A'

    def setup(self) -> Model:
        from bokeh.layouts import column
        return column(Div(text='root'),
                      Div(text='<a href="pool" >pool</a>'))


class Pool(View):

    @property
    def title(self) -> str:
        return 'B'

    def setup(self) -> Model:
        from bokeh.layouts import column
        return column(Div(text='pool'))


if __name__ == '__main__':
    VIEW_TOP = Top()
    VIEW_POOL = Pool()

    BokehServer().start({'/': VIEW_TOP, '/pool': VIEW_POOL})
