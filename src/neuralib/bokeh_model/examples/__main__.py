import argparse

from neuralib.bokeh_model import BokehServer
from .view_all import AllView
from .view_animal import AnimalView
from .view_figure import AnimalFigureView

ap = argparse.ArgumentParser()
ap.add_argument('--animal', metavar='NAME', default=None)
opt = ap.parse_args()

open_url = '/'
if opt.animal is not None:
    open_url = f'/pool?animal={opt.animal}'

BokehServer().start({
    '/': AllView(),
    '/pool': AnimalView(),
    '/pool/figure': AnimalFigureView(),
}, open_url=open_url)
