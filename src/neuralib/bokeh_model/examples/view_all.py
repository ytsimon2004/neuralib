from html import escape
from typing import Optional

from bokeh.model import Model
from bokeh.models import Select, Div

from neuralib.bokeh_model import BokehServer, View
from .model import *

__all__ = ['AllView']


class AllView(View):
    goto_btn: Div
    content: Div
    select_animal: Select
    select_date: Select

    def __init__(self, single=False):
        """

        :param single: disable jump to pool page.
        """
        self.single = single

    @property
    def title(self) -> str:
        return 'ALL'

    def setup(self) -> Model:
        animals = list_animals()
        self.select_animal = Select(
            title='Animal',
            value='',
            options=['', *animals]
        )
        self.select_animal.on_change('value', self.on_select_animal)

        self.select_date = Select(
            title='Date',
            value='',
            options=[]
        )
        self.select_date.on_change('value', self.on_select_date)

        self.goto_btn = Div(text='')

        self.content = Div(text='')

        from bokeh.layouts import row, column
        return column(
            row(
                self.select_animal,
                self.select_date,
                self.goto_btn,
            ),
            self.content
        )

    def update(self):
        self.build_goto_link(None)

    def on_select_animal(self, attr: str, old: str, value: str):
        if len(value):
            self.select_date.options = ['', *list_date(value)]
            self.build_goto_link(value)
        else:
            self.build_goto_link(None)

    def on_select_date(self, attr: str, old: str, value: str):
        if len(value):
            animal = self.select_animal.value
            self.content.text = f'show animal {animal} on date {value}'

    def build_goto_link(self, animal: Optional[str]):
        if animal is None or self.single:
            self.goto_btn.text = f"""
<a type=button 
    class="bk bk-input"
>(Select one animal)</a>
"""
        else:
            self.goto_btn.text = f"""
<a type=button 
    class="bk bk-input"
    href="/pool?animal={animal}"
>{escape(animal)}</a>
"""


def main():
    BokehServer().start(AllView(single=True))


if __name__ == '__main__':
    main()
