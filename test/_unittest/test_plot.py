import unittest
from typing import Callable
from unittest.mock import patch

from matplotlib import pyplot as plt

from _manual.test_plot import (
    test_axes_extend_helper_x,
    test_axes_extend_helper_y,
    test_axes_extend_helper_hist,
    test_axes_extend_helper_bar,
    test_dotplot,
    test_dotplot_ax,
    test_venn2,
    test_venn3
)


class TestPlotting(unittest.TestCase):

    @patch('matplotlib.pyplot.show')
    def plt_close(self, f: Callable, mock_show, *args, **kwargs):
        try:
            f(*args, **kwargs)
            plt.clf()
            plt.close('all')
        except Exception as e:
            self.fail(f'Plotting function raised an exception: {e}')

    def test_plotting_func_runs(self):
        self.plt_close(test_axes_extend_helper_x)
        self.plt_close(test_axes_extend_helper_y)
        self.plt_close(test_axes_extend_helper_hist)
        self.plt_close(test_axes_extend_helper_bar)
        self.plt_close(test_dotplot)
        self.plt_close(test_dotplot_ax)
        self.plt_close(test_venn2)
        self.plt_close(test_venn3)


if __name__ == '__main__':
    unittest.main()
