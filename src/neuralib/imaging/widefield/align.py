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

    color: bool = argument(
        '--color',
        help='alignment map with color',
    )

    def run(self):
        viewer = napari.Viewer()
        cortical_map = load_example_dorsal_cortex(color=self.color, cached=True)

        viewer.open(self.sequence_path)
        viewer.add_image(cortical_map)
        napari.run()


if __name__ == '__main__':
    NapariAlignmentOptions().main()
