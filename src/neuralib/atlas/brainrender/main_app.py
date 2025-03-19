from neuralib.argp import parse_command_args
from neuralib.atlas.brainrender import BrainReconstructor, RoisReconstructor, ProbeReconstructor


def main():
    parsers = dict(
        area=BrainReconstructor,
        roi=RoisReconstructor,
        probe=ProbeReconstructor
    )

    parse_command_args(
        parsers=parsers,
        description='brainrender view',
        usage="""
        Example: 
        >> neuralib_brainrender area -R VISal,VISam,VISl,VISli,VISp,VISpl,VISpm,VISpor
        >> neuralib_brainrender roi -F <FILE>
        >> neuralib_brainrender probe -F <FILE> --depth <DEPTH in um>
        """
    )


if __name__ == '__main__':
    main()
