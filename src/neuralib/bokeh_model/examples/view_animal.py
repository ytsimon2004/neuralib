from typing import Optional, Union

from bokeh.model import Model
from bokeh.models import Div, Select

from neuralib.bokeh_model import BokehServer, View
from .view_figure import AnimalFigureView

__all__ = ['AnimalView']


class AnimalView(View):
    select: Select
    content: Div

    def __init__(self, animal: str = None):
        self._animal: Union[None, str, BaseException] = animal

    @property
    def title(self) -> str:
        animal = self.current_animal or 'Unknown'
        return f'Animal {animal}'

    @property
    def current_animal(self) -> Optional[str]:
        if self._animal is None:
            try:
                self._animal = self.get_arg('animal')[0]
            except (KeyError, IOError) as e:
                self._animal = e

        if isinstance(self._animal, str):
            return self._animal
        else:
            return None

    def setup(self) -> Model:
        self.select = Select(
            title='Figure Type',
            value='',
            options=['', *AnimalFigureView.FIGURE_TYPE]
        )
        self.select.on_change('value', self.on_figure_type)

        self.content = Div(text="", css_classes=['yw-animal-content'])

        from bokeh.layouts import column
        return column(
            self.select,
            self.content,
            Div(text="""
<style>
div.yw-animal-content div {
    position: fixed;
    width: 90%;
    height: 100%;
}
div.yw-animal-content div iframe.bk {
    display: block;
    width: 100%;
    height: 100%;
}
</style>
""")
        )

    def on_figure_type(self, attr: str, old: str, value: str):
        if len(value) == 0:
            self.content.text = ''
            return

        animal = self.current_animal
        if animal is None:
            return

        self.content.text = f""" <iframe src="/pool/figure?animal={animal}&figure={value}" class="bk" ></iframe> """


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--animal', metavar='NAME', required=True)
    opt = ap.parse_args()
    BokehServer().start(AnimalView(opt.animal))


if __name__ == '__main__':
    main()
