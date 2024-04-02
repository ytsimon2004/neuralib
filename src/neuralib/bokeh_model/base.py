"""
Bokeh Base module
=================

:author:
    Ta-Shun Su

[Bokeh](https://docs.bokeh.org/en/latest/) base module for building bokeh application
which is web-based, interactive and python communicate-able.

Simple Example
--------------

>>> from neuralib.bokeh_model import BokehServer, View
>>> class Top(View):
...     def setup(self):
...         from bokeh.models import Div
...         from bokeh.layouts import column
...         return column(Div(text="Hello World!"))
>>> SERVER = BokehServer('Example')
>>> SERVER.start(Top())

General Application Structure
-----------------------------

This structure proposol was following MVC common structure. For this module. **M (Model)** is defined
as a group of functions that handle the data IO and processing. **V (View)** is defined as a group of
classes that hold the bokeh UI components. **C (Controller)** is defined as a group of functions that
bride the Model and View. They are UI event listeners, application controler and other kinds of
function that nothing relate to the data.



>>> # data IO functions
... def list_data() -> list[str]: ...
>>> # Custom ViewComponent
... class DataView(ViewComponent): ...
>>> # Top View
... class Top(View):
...     view_data: DataView
...     def setup(self):
...         # init view_data
...     def update(self):
...         # update view_data
...     # controller functions
...     def on_data_update(self, attr, old, value): ... # property change/update event
...     def on_data_select(self, e): ... # action event
>>> # create cli parser (optional)
>>> # main function (optional)
>>> # create server


"""

import abc
import functools
from typing import Callable, ClassVar, cast, Union, Optional

from bokeh.application import Application
from bokeh.document import Document
from bokeh.models import Model, GlyphRenderer
from bokeh.plotting import figure as _fig
from bokeh.server.server import Server

__all__ = ['Figure', 'View', 'ViewComponent', 'BokehServer']


def Figure(**kwargs):
    return _fig(**kwargs, output_backend='svg')


class View(metaclass=abc.ABCMeta):
    """Top View of bokeh application.

    Example
    -------

    >>> class Top(View):
    ...     # custom viewer, it often follows a Figure
    ...     view_a: ViewA # class ViewA(ViewComponent)
    ...     figure_a: Figure
    ...
    ...     # exported UI components, put here if it needs to be access by controller functions
    ...     select_a: Select
    ...
    ...     def setup(self):
    ...         # initialize custom viewer
    ...         self.figure_a = Figure(...)
    ...         self.view_a = ViewA()
    ...         self.view_a.plot(self.figure_a)
    ...
    ...         # initialize exported UI components
    ...         self.select_a = Select(...)
    ...         self.select_a.on_change('value', on_select_a) # bind 'value' change event listener
    ...
    ...         # local UI components
    ...         # for those simple UIs without complex interaction, such as Button
    ...         button_a = Button('A')
    ...         button_a.on_click(on_btn_a) # bind pressed event listener
    ...
    ...         # layout UI components
    ...         from bokeh.layouts import row, column
    ...         return column(
    ...             self.figure_a,
    ...             row(self.select_a, button_a)
    ...         )

    Implement Note
    --------------

    :class:`View` instance should be singleton, which means there is no second instance existed in the
    same application/python runtime. It is a soft rule (which means you can break this rule without
    and error happen), and this class don't have any mechanism to ensure this rule.

    This class don't force the :meth:`__init__` function signature, which means you can define
    your own `__init__` function with custom parameters, such as arguments from the CLI.
    
    Beside top level updating function :func:`update`, you can define your own updating 
    functions for only smaller group of UI components.

    You can declare all the UI components inside the View class, but you probably will get
    a mess code. Hence, :class:`ViewComponent` is used to group related functions together, included
    some controller functions and data processing functions. By this way, such as variable `view_a`
    in the example, it also provides a namespace that make the code more readable.

    """

    document: Document

    @property
    def title(self) -> str:
        return type(self).__name__

    def get_arg(self, key: str) -> list[str]:
        return list(map(bytes.decode, self.document.session_context.request.arguments[key]))

    @abc.abstractmethod
    def setup(self) -> Model:
        """setup application top view. This function is called by BokehServer only once when
        the server created and opened the web browser.

        In this function, you need to initialize the UI components and return the layout.
        """
        pass

    def update(self):
        """Top level UI components updating function. This function is called by BokehServer
        when :func:`setup` has done.

        In this function, you need to initialize the data and update the UI components' state.
        """
        pass

    def run_later(self, callback: Callable, *args, **kwargs):
        self.document.add_next_tick_callback(functools.partial(callback, *args, **kwargs))

    def run_timeout(self, delay: int, callback: Callable, *args, **kwargs):
        self.document.add_timeout_callback(functools.partial(callback, *args, **kwargs), delay)

    def run_periodic(self, cycle: int, callback: Callable, *args, **kwargs):
        self.document.add_periodic_callback(functools.partial(callback, *args, **kwargs), cycle)

    def on_message(self, message: str, reset: bool):
        """for logging purpose. i.e., TextAreaInput bokeh widget"""
        pass


class ViewComponent(metaclass=abc.ABCMeta):
    """A UI component that provides certain graph on specific data type.

    General Structure
    -----------------

    >>> class MyView(ViewComponent):
    ...     # for ColumnDataSource attribute's name, there is no hard rule
    ...     data_a: ColumnDataSource
    ...     # for GlyphRenderer attribute's name, it should be named with prefix 'render_'
    ...     render_a: GlyphRenderer
    ...     def __init__(self):
    ...         # initialize an empty data
    ...         self.data_a = ColumnDataSource(data=dict(...))
    ...     def plot(self, figure):
    ...         # plotting data
    ...         self.render_a = figure.plot(self.data_a)
    ...     # function to update the render, data
    ...     def update(self, data):
    ...         # update data
    ...         self.data_a.activity = dict(...)


    Implement Note
    --------------

    I suggest make plotting related properties as @property, because they usually need to update/invalid
    other attributes/properties when it is updated. In order to improve the performance, you don't need
    to update the graph/render for each property update in an event loop. Because most render update functions
    have to process data and set value into :class:`ColumnDataSource`, and :class:`ColumnDataSource` need to
    sync the content with the web browser, it will take many times for every property update. You can mention
    (in document) the caller have to call the render update function after several properties update function.

    """

    @abc.abstractmethod
    def plot(self, figure: Figure, **kwargs):
        """plot data in *figure*.

        :param figure: Figure.
        :param kwargs: plotting parameters.
        """
        pass

    @abc.abstractmethod
    def update(self):
        """update the plot"""
        pass

    def set_visible(self, visible: bool, pattern: str = None):
        """Set the visible state of renders for those name contain *pattern*.
        It is a recursive function that it also update the renders inside the
        :class:`ViewComponent` attributes.

        :param visible:
        :param pattern: str in attribute name
        """
        for name in dir(self):
            render = getattr(self, name)
            if name.startswith('render_'):
                if pattern is None or pattern in name:
                    if isinstance(render, list):
                        for _render in render:
                            cast(GlyphRenderer, _render).visible = visible
                    elif isinstance(render, dict):
                        for _render in render.values():
                            cast(GlyphRenderer, _render).visible = visible
                    else:
                        cast(GlyphRenderer, render).visible = visible
            elif isinstance(render, ViewComponent):
                render.set_visible(visible, pattern)

    def list_renders(self, pattern: str = None, recursive: bool = False) -> list[GlyphRenderer]:
        """list all renders for those name contain *pattern*.

        :param pattern: str in attribute name
        :param recursive: recursive find all renders from :class:`ViewComponent` attributes.
        :return:
        """
        ret = []
        for name in dir(self):
            render = getattr(self, name)
            if name.startswith('render_'):
                if pattern is None or pattern in name:
                    if isinstance(render, list):
                        ret.extend(render)
                    elif isinstance(render, dict):
                        ret.extend(render.values())
                    else:
                        ret.append(render)
            elif isinstance(render, ViewComponent) and recursive:
                ret.extend(render.list_renders(pattern, recursive))
        return ret


class BokehServer:
    """Bokeh application server. It is a singleton class, that only one instance can hold the OS port.

    Implement Note
    --------------

    For now, it is a single web-page application, that don't provide any methods to route to other paths.

    """

    INSTANCE: ClassVar['BokehServer'] = None

    server: Server

    def __init__(self, theme: str = 'dark_minimal'):
        self.theme = theme

    def start(self,
              viewer: Union[View, Application, dict[str, Union[View, Application]]],
              open_url: Optional[str] = '/', **kwargs):
        """start serving.

        :param viewer: top view
        :param open_url
        """
        if BokehServer.INSTANCE is None:
            BokehServer.INSTANCE = self
        else:
            raise RuntimeError('server is running')

        if isinstance(viewer, (View, Application)):
            router = {'/': self.page(viewer)}
        else:
            router = {k: self.page(v) for k, v in viewer.items()}

        self.server = Server(router, num_procs=1, **kwargs)
        self.server.start()

        if open_url is not None:
            self.server.io_loop.add_callback(self.server.show, open_url)

        self.server.run_until_shutdown()

    def page(self, viewer: Union[View, Application]):
        """ """
        if isinstance(viewer, Application):
            return viewer

        def _page(doc: Document):
            self._init_doc(doc)
            viewer.document = doc
            doc.add_root(viewer.setup())
            doc.title = viewer.title
            doc.add_next_tick_callback(viewer.update)

        return _page

    def _init_doc(self, doc: Document):
        doc.theme = self.theme
