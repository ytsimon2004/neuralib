from pathlib import Path

import brainrender
import numpy as np
import polars as pl
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
from brainrender.actors import Points
from typing import Literal
from typing import Self

from argclz import AbstractParser, argument, str_tuple_type, validator
from neuralib.atlas.brainrender.util import get_color
from neuralib.atlas.util import allen_to_brainrender_coord
from neuralib.util.logging import setup_clogger

__all__ = ['BrainRenderCLI']

CAMERA_ANGLE_TYPE = Literal['sagittal', 'sagittal2', 'frontal', 'top', 'top_side', 'three_quarters']
SHADER_STYLE_TYPE = Literal['metallic', 'cartoon', 'plastic', 'shiny', 'glossy']


class BrainRenderCLI(AbstractParser):
    """Reconstruct a 3D brain view used brainrender module"""
    DESCRIPTION = 'Reconstruct a 3D brain view used brainrender module'

    DEFAULT_REGION_COLORS = ['lightblue', 'pink', 'turquoise']

    # ============== #
    # BASIC SETTINGS #
    # ============== #

    GROUP_SETTINGS = 'Basic Settings Option'

    camera_angle: CAMERA_ANGLE_TYPE = argument(
        '--camera',
        group=GROUP_SETTINGS,
        default='three_quarters',
        help='camera angle'
    )

    shader_style: SHADER_STYLE_TYPE = argument(
        '--style',
        default='plastic',
        group=GROUP_SETTINGS,
        help='Shader style to use'
    )

    title: str | None = argument(
        '--title',
        metavar='TEXT',
        default=None,
        group=GROUP_SETTINGS,
        help='title added to the top of the window'
    )

    source: str = argument(
        '-S', '--source',
        metavar='NAME',
        default='allen_mouse_10um',
        group=GROUP_SETTINGS,
        help='atlas source name. allen_human_500um as human'
    )

    root_alpha: float = argument(
        '--root-alpha',
        default=0.35,
        group=GROUP_SETTINGS,
        help='root alpha'
    )

    no_root: bool = argument(
        '--no-root',
        group=GROUP_SETTINGS,
        help='render without root(brain) mesh'
    )

    background: Literal['white', 'black'] = argument(
        '--bg',
        default='white',
        group=GROUP_SETTINGS,
        help='background color'
    )

    coordinate_space: Literal['ccf', 'brainrender'] = argument(
        '--coord-space',
        default='ccf',
        group=GROUP_SETTINGS,
        help='which coordinate space, by default ccf'
    )

    # ============= #
    # OPTIONAL VIEW #
    # ============= #

    GROUP_OPTIONAL = 'Optional Option'

    annotation: tuple[str, ...] | None = argument(
        '--annotation',
        validator.tuple().on_item(None, validator.str.match(r'^[^;]+:[^;]+:[^;]+$')) | validator.optional(),
        metavar='AV,DV,ML',
        type=str_tuple_type,
        default=None,
        group=GROUP_OPTIONAL,
        help='whether draw point annotation. e.g., 1.5:1:0.4,-3.2:0.8:0.4 for two points'
    )

    # ============ #
    # GROUP_REGION #
    # ============ #

    GROUP_REGION = 'Region Option'

    regions: str | tuple[str, ...] = argument(
        '-R', '--region',
        metavar='NAME,...',
        type=str_tuple_type,
        default=(),
        group=GROUP_REGION,
        help='region(s) name'
    )

    region_colors: str | tuple[str, ...] | None = argument(
        '--region-color',
        metavar='COLOR,...',
        type=str_tuple_type,
        default=None,
        group=GROUP_REGION,
        help='region(s) color'
    )

    regions_alpha: float = argument(
        '--region-alpha',
        validator.float.in_range_closed(0, 1),
        default=0.35,
        group=GROUP_REGION,
        help='region alpha value'
    )

    hemisphere: Literal['right', 'left', 'both'] = argument(
        '-H', '--hemisphere',
        default='both',
        group=GROUP_REGION,
        help='which hemisphere for rendering the region'
    )

    print_tree: bool = argument(
        '--print-tree',
        group=GROUP_REGION,
        help='print tree for the available regions for the given source'
    )

    tree_init: str | None = argument(
        '--tree-init',
        default=None,
        group=GROUP_REGION,
        help='init region for the tree print'
    )

    print_name: bool = argument(
        '--print-name',
        group=GROUP_REGION,
        help='print acronym and the corresponding name'
    )

    # ============ #
    # GROUP_OUTPUT #
    # ============ #

    GROUP_OUTPUT = 'Output Option'

    video_output: Path | None = argument(
        '--video-output',
        validator.path.is_suffix(['.mp4', '.avi']).optional(),
        default=None,
        group=GROUP_OUTPUT,
        help='video output path'
    )

    output: Path | None = argument(
        '-O', '--output',
        validator.path.is_suffix('.html').optional(),
        default=None,
        group=GROUP_OUTPUT,
        help='output path for the html, if None, preview'
    )

    #
    scene: brainrender.Scene
    logger = setup_clogger()
    _stop_render: bool = False  # flag for print mode

    def post_parsing(self):
        self._render_settings()
        self._verbose()

    def _render_settings(self):
        from brainrender import settings
        settings.BACKGROUND_COLOR = self.background
        settings.DEFAULT_ATLAS = self.source
        settings.ROOT_ALPHA = self.root_alpha
        settings.SHOW_AXES = False
        settings.WHOLE_SCREEN = False
        settings.DEFAULT_CAMERA = self.camera_angle
        settings.SHADER_STYLE = self.shader_style

        settings.vsettings.screenshot_transparent_background = True
        settings.vsettings.use_fxaa = False

    def _verbose(self):
        if self.print_tree:
            from neuralib.atlas.plot import plot_structure_tree
            plot_structure_tree(self.tree_init)
            self._stop_render = True

        if self.print_name:
            from neuralib.util.table import rich_data_frame_table
            file = self.get_atlas_brain_globe().root_dir / 'structures.csv'
            df = pl.read_csv(file).select('acronym', 'name')
            rich_data_frame_table(df)
            self._stop_render = True

    def run(self):
        self.post_parsing()

        if not self._stop_render:
            self.render()
            self.render_output()

    def render(self):
        """brainrender interactive"""
        self.scene = brainrender.Scene(root=not self.no_root, inset=False, title=self.title, screenshots_folder='.')
        self.scene.plotter.camera.Zoom(0.3)

        if self.annotation is not None:
            self._reconstruct_annotation()

        self._reconstruct_region()

    def render_output(self):
        """io handling. i.e., video, html output"""
        if self.video_output is not None:
            self.source = 'allen_mouse_25um'  # force set for whole brain scene
            self.video_maker(self.video_output)
        elif self.output is not None:
            self.export(self)
        else:
            self.scene.render()

    def _reconstruct_annotation(self):
        for ann in self.annotation:
            ap, dv, ml = tuple(map(float, ann.split(':')))
            dat = allen_to_brainrender_coord(np.array([ap, dv, ml]))  # (N, 3)
            self.scene.add(Points(dat, radius=120))

    def _reconstruct_region(self):
        color_list = self.region_colors or self.DEFAULT_REGION_COLORS

        if len(self.regions) != 0:
            for i, region in enumerate(self.regions):
                try:
                    color = color_list[i]
                except IndexError:
                    color = get_color(i, [''])

                self.logger.info(f'Plot Rois File: {i}, {region}, {color}')
                self.scene.add_brain_region(region, color=color, alpha=self.regions_alpha, hemisphere=self.hemisphere)

    @classmethod
    def export(cls, reconstructor: Self | None,
               output: Path | None = None,
               areas: list[str] | None = None,
               alpha: float = 0.15):
        """
        Export reconstruction as html

        :param reconstructor: `BrainRenderReconstructor` if use the current scene, and --output cli.
                    Otherwise, general func usage
        :param output: output file path
        :param areas: list of area(s)
        :param alpha: brain region alpha
        """

        if isinstance(reconstructor, BrainRenderCLI):
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
        """
        generate video

        :param output_file: video output path
        """
        from brainrender import VideoMaker

        d, f = output_file.parent, output_file.stem
        vm = VideoMaker(self.scene, save_fld=d, name=f)
        vm.make_video(azimuth=1, elevation=0, roll=0)

    def get_atlas_brain_globe(self, check_latest=False) -> BrainGlobeAtlas:
        return BrainGlobeAtlas(
            self.source,
            check_latest=check_latest
        )


if __name__ == '__main__':
    BrainRenderCLI().main()
