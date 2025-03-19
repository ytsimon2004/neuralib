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
        Usage Examples:
    
        Render a brain region:
        $ neuralib_brainrender area -R <REGION, ...>
        $ neuralib_brainrender area -R VISal,VISam,VISl,VISli,VISp,VISpl,VISpm,VISpor
    
        Render a region of interest (ROI) from a file:
        $ neuralib_brainrender roi -F <FILE>
    
        Render a probe placement from a file with depth specification:
        $ neuralib_brainrender probe -F <FILE> --depth <DEPTH>
        """
    )


if __name__ == '__main__':
    main()
