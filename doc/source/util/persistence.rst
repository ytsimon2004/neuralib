Persistence Class
=================

This module defines the `persistence_class` decorator for managing structured persistence
of data classes, along with optional integration with caching and command-line persistence tools.

- **Refer to API:** :mod:`neuralib.persistence`

Define a Persistence Class
--------------------------

.. code-block:: python

    import numpy as np
    from neuralib import persistence

    @persistence.persistence_class
    class Example:
        # Key fields — use `filename=True` to include in filename generation
        use_animal: str = persistence.field(validator=True, filename=True)
        use_session: str = persistence.field(validator=True)
        use_date: str = persistence.field(validator=True, filename=True)

        # Data fields
        channels: list[int]
        data: np.ndarray

Load / Save
^^^^^^^^^^^

.. code-block:: python

    example = Example(use_animal='A00', use_session='', use_date='1234')
    save(example, 'example.pkl')
    example_2 = load(Example, 'example.pkl')  # Should be content-identical to `example`

Cooperate with PersistenceOptions
---------------------------------

.. code-block:: python

    from neuralib.persistence.cli_persistence import PersistenceOptions

    class ExampleHandle(PersistenceOptions[Example]):
        def empty_cache(self) -> Example:
            return Example(use_animal='A00', use_session='', use_date='1234')

        def compute_cache(self, result: Example) -> Example:
            result.channels = [0, 1, 2]
            result.riglog = np.array(result.channels)
            return result

Dynamically Generated Methods
-----------------------------

Persistence classes generate several methods automatically:

1. ``__init__`` (with parameters from all `persistence.field` definitions)

   .. code-block:: python

       def __init__(self, use_animal: str, use_session: str, use_date: str):
           ...

2. ``__str__`` returns filename

   .. code-block:: python

       def __str__(self):
           return filename(self)

3. ``__repr__`` prints internal fields

   .. code-block:: python

       def __repr__(self):
           return 'Example{' + f'use_animal={self.use_animal}, use_session={self.use_session}, use_date={self.use_date}' + '}'

4. ``_replace`` behaves like ``NamedTuple._replace`` when a stub ``_replace`` is defined.

   .. code-block:: python

       def _replace(self, *, use_animal=missing, use_session=missing, use_date=missing, channels=missing, data=missing) -> Example:
           ...

Auto-Increment Field
--------------------

To handle saving multiple results from the same data source (e.g., randomized or shuffled outputs), you can use an auto-incrementing field via :func:`autoinc_field`.

.. code-block:: python

    @persistence.persistence_class
    class Result:
        a: str = persistence.field(validator=True, filename=True)
        b: int = persistence.autoinc_field()
        c: str

        def __init__(self, a: str, b: int = None):
            ...

        def _replace(self, *, a: str, c: str):
            ...

Auto-increment rules:

1. Only **one** `autoinc_field` is allowed per class.
2. It must be of type `int`.
3. Loading fails if autoinc value is unresolved.
4. On saving, autoinc is resolved to `max(existing) + 1`.

Pickle Format
-------------

Persistence classes are saved using `as_dict` conversion and pickled.

**IMPORTANT:**
If your class defines a custom `__init__` (not matching the auto-generated one), you **must** also define a `from_dict` method:

.. code-block:: python

    @persistence.persistence_class
    class Example:
        a: int = persistence.field(validator=True, filename=True)
        b: int = persistence.field(validator=True, filename=True)
        c: int

        def __init__(self):
            ...

        @classmethod
        def from_dict(cls, data: dict[str, Any]) -> 'Example':
            # reconstruct the instance from the dictionary
            ...
