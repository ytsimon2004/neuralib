from pathlib import Path

from bokeh.models import ColorBar, LinearColorMapper, GlyphRenderer, TextAreaInput, Div
from bokeh.plotting import figure

__all__ = ['ColorBarView',
           'MsgLog',
           'change_html_format',
           'add_html_format']


class ColorBarView:

    def __init__(self,
                 render: GlyphRenderer,
                 key: str,
                 palette: str,
                 **kwargs):
        """
        Create color map
        :param render: `GlyphRenderer` render
        :param key: key in the `ColumnDataSource`, which indicates the value changes
        :param palette:
        """
        self.render = render
        self.key = key
        self.palette = palette

        self.color_mapper = LinearColorMapper(palette=self.palette, domain=[(self.render, f'{self.key}')])
        self.color_bar = ColorBar(color_mapper=self.color_mapper, width=10, label_standoff=12, **kwargs)

    def insert(self, fig: figure):
        fig.add_layout(self.color_bar, 'right')


def insert_color_bar(fig: figure, render: GlyphRenderer, key: str, palette: str, **kwargs):
    """

    :param fig:
    :param render:
    :param key:
    :param palette:
    :param kwargs:
    :return:
    """
    color_mapper = LinearColorMapper(palette=palette, domain=[(render, key)])
    color_bar = ColorBar(color_mapper=color_mapper, width=10, label_standoff=12, **kwargs)
    fig.add_layout(color_bar)


class MsgLog:
    error_set: bool  # flag if error msg

    def __init__(self, value: str, **kwargs):
        self.error_set = False
        self.message_area = TextAreaInput(title='Log:',
                                          value=value,
                                          rows=5,
                                          cols=100,
                                          disabled=True,
                                          css_classes=['message-area'],
                                          **kwargs)

    def on_message(self, message: str, reset: bool):
        """
        *Example*

        >>> class View(View):
        ...   INSTANCE = None
        ...   def __init__(self, *arg, **kwargs):
        ...         View.INSTANCE = self

        >>> def msg_log(*message: str, reset: bool = False)
        ...     View.INSTANCE.msg_log.on_message(message, reset)

        :param message:
        :param reset: If true, reset the mesasge log `TextAreaInput`
        :return:
        """
        self.message_area.disabled = False
        text = self.message_area.value

        if 'ERR' in message:
            self.error_set = True
        elif reset:
            self.error_set = False

        if self.error_set:
            self.message_area.css_classes = ['message-area-error-msg']
        else:
            self.message_area.css_classes = ['message-area']

        if reset:
            self.message_area.value = str(message)
        else:
            self.message_area.value = text + '\n' + str(message)

        self.message_area.disabled = True

    @property
    def set_div(self) -> Div:
        return Div(text="""
            <style type="text/css">
            div.message-area textarea.bk.bk-input {
                background-color: black !important;
                color: lightgreen !important;
                font-family: monospace !important;
            }
            div.message-area-error-msg textarea.bk.bk-input {
                background-color: black !important;
                color: red !important;
                font-family: monospace !important;
            }                
            div.message-area label.bk {
                       color: lightgreen !important;
                       font-family: monospace !important
                   }
            </style> 
            """)


def change_html_format(input_file: Path, *style: dict[str, str]):
    """change style im embedded html in bokeh with `existing` <style> header"""

    tmp = input_file.with_name('tmp.txt')
    with open(input_file, 'r') as f:
        with tmp.open('w') as out:
            for i, line in enumerate(f):
                for s in style:
                    key = list(s.keys())[0]
                    value = list(s.values())[0]
                    if line.strip().startswith(f'{key}'):
                        out.write(line)
                        out.write(f'\t\t\t{value};\n')
                    else:
                        out.write(line)

    tmp.rename(input_file)


def add_html_format(input_file: Path, style: dict[str, str]):
    """change style im embedded html in bokeh with `new` <style> header"""
    tmp = input_file.with_name('tmp.txt')
    with open(input_file, 'r') as f:
        with tmp.open('w') as out:
            for i, line in enumerate(f):
                if line.strip().startswith('<style>'):
                    key = list(style.keys())[0]
                    value = list(style.values())[0]
                    out.write(line)
                    out.write('\t\t{0} {{\n\t\t\t{1}\n\t\t;}}\n\n'.format(key, value))
                else:
                    out.write(line)

    tmp.rename(input_file)
