import numpy as np

from neuralib.plot import plot_figure
from neuralib.plot.colormap import insert_colorbar, insert_cyclic_colorbar
from neuralib.typing import PathLike
from .fft import SequenceFFT

__all__ = ['plot_retinotopic_maps']


def plot_retinotopic_maps(sequence: np.ndarray, *,
                          output: PathLike | None,
                          interp: str = 'none',
                          intensity_cmap='binary',
                          phase_cmap='hsv',
                          **kwargs):
    """
    Plot retinotopic maps based on fft calculation.

    .. seealso::

        :class:`~neuralib.imaging.widefield.fft.SequenceFFT`

    :param sequence: Image sequence. `Array[float | uint8, [F, H, W]]`
    :param output: Output path for the figure, defaults is None for ``show()``
    :param interp: Kwarg interpolation for ``ax.imshow()``
    :param intensity_cmap: Intensity color map, defaults to 'binary'
    :param phase_cmap: Intensity phase color map, defaults to 'hsv'
    :param kwargs: Additional arguments passed to :meth:`~neuralib.imaging.widefield.fft.SequenceFFT.as_colormap()`
    """
    seq_fft = SequenceFFT(sequence)

    with plot_figure(output, 1, 3, figsize=(12, 6)) as _ax:
        ax = _ax[0]
        im = ax.imshow(seq_fft.get_intensity(), cmap=intensity_cmap, interpolation=interp)
        insert_colorbar(ax, im)
        ax.set_title('intensity')
        ax.axis('off')

        ax = _ax[1]
        im = ax.imshow(seq_fft.get_phase(), cmap=phase_cmap, interpolation=interp)
        insert_cyclic_colorbar(ax, im, num_colors=36, width=0.2, inner_diameter=1, vmin=0, vmax=1)
        ax.set_title('phase')
        ax.axis('off')

        ax = _ax[2]
        im = ax.imshow(seq_fft.as_colormap(**kwargs), cmap='hsv', interpolation=interp)
        insert_cyclic_colorbar(ax, im, num_colors=36, width=0.5, inner_diameter=1, vmin=0, vmax=1)
        ax.set_title('retinotopic map')
        ax.axis('off')
