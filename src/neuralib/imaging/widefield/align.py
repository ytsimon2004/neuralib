import napari

from neuralib.argp import AbstractParser, argument
from neuralib.io.dataset import load_example_dorsal_cortex

__all__ = ['NapariAlignmentOptions']


class NapariAlignmentOptions(AbstractParser):
    DESCRIPTION = 'View the video sequences and top view of dorsal cortex using napari'

    sequence_path: str = argument(
        metavar="FILE",
        help='file path for the video sequence'
    )

    reference_path: str | None = argument(
        '-R', '--reference',
        default=None,
        help='reference image (e.g., bright-field)'
    )

    dorsal_map_path: str | None = argument(
        '-M', '--map',
        default=None,
        help='dorsal map file, If None then use default'
    )

    color: bool = argument(
        '--color',
        help='alignment map with color',
    )

    def run(self):
        viewer = napari.Viewer()
        viewer.open(self.sequence_path)

        #
        if self.dorsal_map_path is not None:
            viewer.open(self.dorsal_map_path)
        else:
            cortical_map = load_example_dorsal_cortex(color=self.color, cached=True)
            viewer.add_image(cortical_map)

        #
        if self.reference_path is not None:
            viewer.open(self.reference_path)

        napari.run()


if __name__ == '__main__':
    NapariAlignmentOptions().main()
