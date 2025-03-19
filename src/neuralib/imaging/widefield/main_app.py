from neuralib.argp import parse_command_args

from .fft_view import WideFieldFFTViewOption


def main():
    parsers = dict(
        fft=WideFieldFFTViewOption
    )

    parse_command_args(
        parsers=parsers,
        description='widefield tools',
        usage="""
        Usage Examples:

        View HSV map in FFT:
        $ neuralib_widefield fft -F <FILE>
        
        Alignment napari for image sequence
        $ neuralib_widefield align -F <FILE>
        
        """
    )


if __name__ == '__main__':
    main()
