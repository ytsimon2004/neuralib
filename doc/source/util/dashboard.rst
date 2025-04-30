Bokeh Base Module
=================

`Bokeh <https://docs.bokeh.org/en/latest/>`_ base module for building Bokeh applications
that are web-based, interactive, and can communicate with Python backends.

- **Refer to API**: :mod:`neuralib.dashboard`

Simple Example
--------------

.. code-block:: python

    from neuralib.dashboard import BokehServer, View

    class Top(View):
        def setup(self):
            from bokeh.models import Div
            from bokeh.layouts import column
            return column(Div(text="Hello World!"))

    SERVER = BokehServer('Example')
    SERVER.start(Top())

General Application Structure
-----------------------------

This structure follows the common **MVC** (Model-View-Controller) architecture:

- **Model (M)**: A group of functions that handle data I/O and processing.
- **View (V)**: A group of classes that hold Bokeh UI components.
- **Controller (C)**: A group of functions that bridge the Model and View. These include UI event listeners,
  application control logic, and functions unrelated to direct data manipulation.

.. code-block:: python

    # Data I/O functions
    def list_data() -> list[str]:
        ...

    # Custom ViewComponent
    class DataView(ViewComponent):
        ...

    # Top View
    class Top(View):
        view_data: DataView

        def setup(self):
            # Initialize view_data
            ...

        def update(self):
            # Update view_data
            ...

        # Controller functions
        def on_data_update(self, attr, old, value):
            ...  # Property change/update event

        def on_data_select(self, e):
            ...  # Action event

    # Optional: CLI parser, main function, and server creation

