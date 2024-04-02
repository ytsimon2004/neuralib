from typing import Union, Optional

from bokeh.model import Model
from bokeh.models import Div

from neuralib.bokeh_model import View
from .model import list_date

__all__ = ['AnimalFigureView']


class AnimalFigureView(View):
    FIGURE_TYPE = ["default", 'other', 'more']
    content_title: Div
    content_date: Div

    def __init__(self, animal: str = None, figure: str = None):
        self._animal: Union[None, str, BaseException] = animal
        self._figure: Optional[str] = figure

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

    @property
    def current_figure(self) -> str:
        if self._figure is None:
            try:
                self._figure = self.get_arg('figure')[0]
            except (KeyError, IOError) as e:
                self._figure = 'default'
        return self._figure

    def setup(self) -> Model:
        self.content_title = Div(text=f"animal {self.current_animal} with figure {self.current_figure}")
        self.content_date = Div(text="Date:<ol>" + "\n".join([
            f"<li>{date}</li>" for date in list_date(self.current_animal)
        ]) + "</ol>")

        from bokeh.layouts import column
        return column(
            self.content_title,
            self.content_date,
        )
