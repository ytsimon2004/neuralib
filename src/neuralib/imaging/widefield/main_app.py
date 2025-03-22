from neuralib.argp import parse_command_args
from neuralib.imaging.widefield.align import NapariAlignmentOptions

from .fft_view import WideFieldFFTViewOption


def main():
    parsers = dict(
        align=NapariAlignmentOptions,
        fft=WideFieldFFTViewOption
    )

    parse_command_args(
        parsers=parsers,
        description='widefield tools',
        usage="""
        Usage Examples:

        View HSV map in FFT:
        $ neuralib_widefield fft <FILE>
        
        Alignment napari for image sequence
        $ neuralib_widefield align <FILE>
        
        """
    )


if __name__ == '__main__':
    main()
