import numpy as np
from iblatlas.plots import prepare_lr_data

from neuralib.atlas.ibl.plot import IBLAtlasPlotWrapper


def plot_single_hemisphere(ibl: IBLAtlasPlotWrapper):
    acronyms = np.array(['VPM', 'VPL', 'PO', 'LP', 'CA1', 'DG-mo'])
    ibl.plot_scalar_on_slice(acronyms, coord=-2000, plane='coronal', background='boundary', cmap='Reds')


def plot_automerge(ibl: IBLAtlasPlotWrapper):
    acronyms = np.array(['RSPagl', 'RSPd', 'RSPv'])
    ibl.plot_scalar_on_slice(acronyms, coord=-2000, plane='coronal', background='image', cmap='Reds', mapping='Beryl')


def plot_both_hemisphere(ibl: IBLAtlasPlotWrapper):
    acronyms_lh = np.array(['LP', 'CA1'])
    values_lh = np.random.randint(0, 10, acronyms_lh.size)
    acronyms_rh = np.array(['DG-mo', 'VISa5'])
    values_rh = np.random.randint(0, 10, acronyms_rh.size)
    acronyms_lr, values_lr = prepare_lr_data(acronyms_lh, values_lh, acronyms_rh, values_rh)
    ibl.plot_scalar_on_slice(acronyms_lr, values=values_lr,
                             coord=-1800, background='boundary', hemisphere='both')


def plot_sagittal_automerge(ibl: IBLAtlasPlotWrapper):
    acronyms = np.array(['VPM', 'VPL', 'PO', 'LP', 'CA1', 'DG-mo', 'SSs5', 'VISa5', 'AUDv6a', 'MOp5', 'FRP5'])
    ibl.plot_scalar_on_slice(acronyms,
                             coord=-2000, plane='sagittal', mapping='Cosmos',
                             hemisphere='left', background='image', cmap='Greens')


def plot_horizontal(ibl: IBLAtlasPlotWrapper):
    acronyms_lh = np.array(['LP', 'CA1'])
    values_lh = np.random.randint(0, 10, acronyms_lh.size)
    acronyms_rh = np.array(['DG-mo', 'VISa5'])
    values_rh = np.random.randint(0, 10, acronyms_rh.size)
    acronyms_lr, values_lr = prepare_lr_data(acronyms_lh, values_lh, acronyms_rh, values_rh)
    ibl.plot_scalar_on_slice(acronyms_lr, values=values_lr, coord=-2500, plane='horizontal', mapping='Allen',
                             hemisphere='both', background='image', cmap='Reds', clevels=[0, 5])


def plot_top_view(ibl: IBLAtlasPlotWrapper, **kwargs):
    acronyms = np.array(['RSPagl', 'RSPd', 'RSPv'])
    ibl.plot_scalar_on_slice(acronyms, coord=-2000, plane='top', hemisphere='left',
                             background='boundary', cmap='Purples', mapping='Beryl', **kwargs)
