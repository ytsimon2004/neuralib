"""
Wrapper for iblatlas
======================

Provide basic plotting usage

See detail in ``iblatlas.plots.plot_scalar_on_slice``

.. seealso:: `<https://int-brain-lab.github.io/iblenv/notebooks_external/atlas_plotting_scalar_on_slice.html>`_


Example of plot the allen map with given areas
------------------------------------------------

.. code-block:: python

    import numpy as np
    from iblatlas.plots import prepare_lr_data
    from matplotlib import pyplot as plt
    from neuralib.atlas.ibl.plot import IBLAtlasPlotWrapper

    acronyms = np.array(['VPM', 'VPL', 'PO', 'LP', 'CA1', 'DG-mo'])
    ibl.plot_scalar_on_slice(acronyms, coord=-2000, plane='coronal', background='boundary', cmap='Reds')
    plt.show()


Example of plot the automerged subregions
------------------------------------------------

.. code-block:: python

    acronyms = np.array(['RSPagl', 'RSPd', 'RSPv'])
    # `Beryl` mapping merge
    ibl.plot_scalar_on_slice(acronyms, coord=-2000, plane='coronal', background='image', cmap='Reds', mapping='Beryl')
    plt.show()


Example of plot the sagittal subregions
------------------------------------------------


.. code-block:: python

    acronyms = np.array(['VPM', 'VPL', 'PO', 'LP', 'CA1', 'DG-mo', 'SSs5', 'VISa5', 'AUDv6a', 'MOp5', 'FRP5'])
    ibl.plot_scalar_on_slice(acronyms,
                             coord=-2000, plane='sagittal', mapping='Cosmos',
                             hemisphere='left', background='image', cmap='Greens')

    plt.show()


Example of plot the transverse view
------------------------------------------------

.. code-block:: python

    acronyms_lh = np.array(['LP', 'CA1'])
    values_lh = np.random.randint(0, 10, acronyms_lh.size)
    acronyms_rh = np.array(['DG-mo', 'VISa5'])
    values_rh = np.random.randint(0, 10, acronyms_rh.size)
    acronyms_lr, values_lr = prepare_lr_data(acronyms_lh, values_lh, acronyms_rh, values_rh)
    ibl.plot_scalar_on_slice(acronyms_lr, values=values_lr, coord=-2500, plane='horizontal', mapping='Allen',
                             hemisphere='both', background='image', cmap='Reds', clevels=[0, 5])

     plt.show()


Example of plot the Top view (dorsal cortex)
-----------------------------------------------

.. code-block:: python

    acronyms = np.array(['RSPagl', 'RSPd', 'RSPv'])
    ibl.plot_scalar_on_slice(acronyms, coord=-2000, plane='top', hemisphere='left',
                             background='boundary', cmap='Purples', mapping='Beryl', **kwargs)

    plt.show()






"""

from .plot import *
