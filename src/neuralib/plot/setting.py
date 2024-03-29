from matplotlib.axes import Axes

__all__ = ['ax_log_setting']


def ax_log_setting(ax: Axes, **kwargs):
    """log scale and tick setting"""
    import matplotlib.ticker as mticker

    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_yticks([0.04, 0.08, 0.16])  # hardcode that used for interpolation plotting
    ax.set_xticks([1, 2, 4])  # hardcode that used for interpolation plotting
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')  # no matter xy scaling, set square

    # against the `plot_figure` ctx manager
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    for axis in ['bottom', 'top', 'left', 'right']:
        ax.spines[axis].set_linewidth(1)

    ax.set(**kwargs)
