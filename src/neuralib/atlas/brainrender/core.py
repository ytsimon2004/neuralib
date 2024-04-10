import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Union, Optional, Literal, TypedDict

import brainrender
import numpy as np
from bg_atlasapi import BrainGlobeAtlas
from brainrender.actors import Points
from typing_extensions import TypeAlias

from neuralib.argp import AbstractParser, argument, str_tuple_type
from neuralib.atlas.brainrender.util import get_color
from neuralib.atlas.type import Source
from neuralib.atlas.util import roi_points_converter
from neuralib.util.color_logging import setup_clogger

__all__ = [
    'ROI_COLORS',
    'RoiType',
    'BrainReconstructor',
]

ROI_COLORS = ['purple', 'gold', 'grey']
DEFAULT_REGION_COLORS = ['lightblue', 'pink', 'turquoise']
CAMERA_ANGLE_TYPE = Literal['sagittal', 'sagittal2', 'frontal', 'top', 'top_side', 'three_quarters']
SHADER_STYLE_TYPE = Literal['metallic', 'cartoon', 'plastic', 'shiny', 'glossy']

RoiType: TypeAlias = list[Union[list[np.ndarray], np.ndarray]]

Logger = setup_clogger(caller_name=Path(__file__).name)


class CameraAngle(TypedDict):
    """if customized"""
    pos: tuple[int, int, int]
    viewup: tuple[int, int, int]
    clippingRange: tuple[int, int]


class BrainReconstructor(AbstractParser):
    """Wrapper for brainrender 3D reconstruction"""
    DESCRIPTION = 'reconstruct a 3D brain view used brainrender module'
    EPILOG = """
        Data shape of points.
        (N, 3) floating matrix with row (x, y, z), or (N, 4) with row (x, y, z, r),
        where r indicate the index (start from 1, 0 means non-region) in --regions.
        Correspond points and region will use the same color.
        """

    title: str = argument('-T', '--title', metavar='TEXT', default=None)
    source: str = argument('--source', default='allen_mouse_10um',
                           help='atlas source name. allen_human_500um for human')

    # scene
    no_root: bool = argument('--no-root', help='render without root(brain) mesh')

    # source notation
    with_source: bool = argument('-S', '--src', help='whether draw the location for source (experiment dependent)')

    # points
    csv_file: Optional[Path] = argument('-F', '--file', type=Path, default=None, help='csv file')
    points_file: list[Path] = argument('--points', metavar='FILE', type=Path, default=[], action='append')
    radius: float = argument('--roi-radius', default=30, help='each roi radius')
    output: Path = argument('-O', '--out', default=None, help='output path for the html, if None, preview')

    # region
    regions: str = argument('-R', '--region', metavar='NAME,...', type=str_tuple_type, default=[])
    region_colors: str = argument('-C', '--color', metavar='COLOR,...', type=str_tuple_type, default=None)
    regions_alpha: float = argument('--region-alpha', type=float, default=0.35, help='region alpha')
    hemisphere: Literal['right', 'left', 'both'] = argument('-H', '--hemisphere', default='both',
                                                            help='which hemisphere')

    #
    video_output: Optional[Path] = argument('-V', '--video-output', default=None, help='video output')

    # settings
    background: Literal['white', 'black'] = argument('--bg', default='white', help='background color')
    camera_angle: CAMERA_ANGLE_TYPE = argument('--camera', default='three_quarters')
    shader_style: SHADER_STYLE_TYPE = argument('--style', default='plastic')

    #
    scene: brainrender.Scene

    def __init__(self):
        self._need_close_file: list[NamedTemporaryFile] = []
        self.points_list: list[str] = []

    def _render_settings(self):
        from brainrender import settings
        settings.BACKGROUND_COLOR = self.background
        settings.DEFAULT_ATLAS = self.source
        settings.ROOT_ALPHA = 0.35 if self.background == 'black' else 0.2
        settings.SHOW_AXES = False
        settings.WHOLE_SCREEN = False
        settings.DEFAULT_CAMERA = self.camera_angle
        settings.SHADER_STYLE = self.shader_style

        settings.vsettings.screenshot_transparent_background = True
        settings.vsettings.use_fxaa = False

    def post_parsing(self):
        if self.video_output is not None:
            self.source = 'allen_mouse_25um'  # force set for whole brain scene

        self._render_settings()

    def run(self):
        self.scene = brainrender.Scene(root=not self.no_root, inset=False, title=self.title, screenshots_folder='.')
        self.load()
        self.add_points_from_file()
        self.reconstruct()

        #
        if self.video_output is not None:
            self.video_maker(self.video_output)
        elif self.output is not None:
            self.export(self)
        else:
            print('render...')
            self.scene.render()

        #
        if len(self._need_close_file) != 0:
            for f in self._need_close_file:
                f.close()
                Path(f.name).unlink(missing_ok=True)  # winOS

    @property
    def n_files(self) -> int:
        return len(self.points_file)

    # ====== #
    # Points #
    # ====== #

    def add_points(self, rois_list: RoiType):
        for p in rois_list:
            if isinstance(p, np.ndarray):
                # create temporal file in memory for p
                # os handle for NamedTemporaryFile, https://stackoverflow.com/a/23212515
                if sys.platform == 'win32':
                    delete = False
                else:
                    delete = True
                f = NamedTemporaryFile(prefix='.temp-run-3d-proj-', suffix='.npy', delete=delete)

                np.save(f, p)
                f.seek(0)
                self.points_list.append(f.name)
                self._need_close_file.append(f)

    def add_points_from_file(self):
        if self.n_files != 0:
            for it in self.points_file:
                if it.suffix not in ('npy', 'npz'):
                    raise ValueError(f'invalid suffix: {it.suffix}')
                self.points_list.append(str(it))

    def load(self):
        """Overwrite by children"""
        pass

    # =========== #
    # Reconstruct #
    # =========== #

    def reconstruct(self):
        if self.with_source:
            self._reconstruct_source()

        self._reconstruct_region()
        self._reconstruct_points_from_file()

    def _reconstruct_source(self):
        """Depending on the experimental purpose, i.e., viral injection site, targeted location..."""
        # TODO generalize
        src: dict[Source, np.ndarray] = {
            'aRSC': np.array([-1.5, 1, 0.4]),
            'pRSC': np.array([-3.2, 0.8, 0.4])
        }

        color: dict[Source, str] = {
            'aRSC': 'gold',
            'pRSC': 'violet'
        }

        for source, coords in src.items():
            points = roi_points_converter(coords)
            self.scene.add(Points(points, colors=color[source], radius=120))

    def _reconstruct_region(self):
        if self.region_colors is not None:
            color_list = self.region_colors
            assert len(color_list) == len(self.regions)
        else:
            color_list = DEFAULT_REGION_COLORS

        if len(self.regions) != 0:
            for i, region in enumerate(self.regions):
                if len(self.regions) > len(color_list):
                    color = get_color(i, [''])
                else:
                    color = get_color(i, color_list)

                Logger.info(f'Plot Rois File: {i}, {region}, {color}')
                self.scene.add_brain_region(region, color=color, alpha=self.regions_alpha, hemisphere=self.hemisphere)

    def _reconstruct_points_from_file(self):
        for i, file in enumerate(self.points_list):
            data = np.load(file)

            if data.ndim != 2:
                raise ValueError(f'wrong dimension: {data.shape}')
            #
            if data.shape[1] == 3:
                colors = get_color(i, ROI_COLORS)
                Logger.info(f'Plot Rois File: {i}, {file}, {colors}')
                self.scene.add(Points(data, name='roi', colors=colors, alpha=0.9, radius=self.radius))

            elif data.shape[1] == 4:  # TODO not test yet
                k = data[:, 3].astype(int)
                for t in np.unique(k):
                    self.scene.add(Points(
                        data[k == t, 0:3],
                        name='rois',
                        colors=get_color(t, ROI_COLORS),
                        alpha=0.6,
                        radius=self.radius
                    ))
            else:
                raise ValueError(f'wrong shape: {data.shape}: {file}')

    @classmethod
    def export(cls, reconstructor: Optional['BrainReconstructor'],
               output: Path = None,
               areas: list[str] = None,
               alpha: float = 0.15):
        """
        export reconstruction as html
        TODO check export / view were different hemispheres, seems opposite in export..

        :param reconstructor: `BrainRenderReconstructor` if use the current scene, and --output cli.
                    Otherwise, general func usage
        :param output:
        :param areas: list of area(s)
        :param alpha:
        :return:
        """

        if isinstance(reconstructor, BrainReconstructor):
            scene = reconstructor.scene
            output = reconstructor.output
        else:
            scene = brainrender.Scene(inset=False, title='', screenshots_folder='.')
            output = output

        if areas is not None:
            if not isinstance(areas, list):
                raise TypeError('')

            for it in areas:
                scene.add_brain_region(it, alpha=alpha)

        scene.export(output)

    def video_maker(self, output_file: Path):
        from brainrender import VideoMaker
        import vedo
        #
        vedo.settings.default_backend = 'vtk'

        #
        d, f = output_file.parent, output_file.stem
        vm = VideoMaker(self.scene, save_fld=d, name=f)
        vm.make_video(azimuth=1, elevation=0, roll=0)

    def get_atlas_brain_globe(self, check_latest=False) -> BrainGlobeAtlas:
        return BrainGlobeAtlas(
            self.source,
            check_latest=check_latest
        )


if __name__ == '__main__':
    BrainReconstructor().main()
