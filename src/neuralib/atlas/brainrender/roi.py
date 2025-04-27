import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import polars as pl
from brainrender.actors import Points

from argclz import argument, str_tuple_type, validator, list_type
from neuralib.atlas.brainrender.core import BrainRenderCLI
from neuralib.atlas.brainrender.util import get_color
from neuralib.atlas.util import iter_source_coordinates, allen_to_brainrender_coord, as_coords_array

__all__ = ['RoiRenderCLI']


class RoiRenderCLI(BrainRenderCLI):
    """ROIs reconstruction with brainrender"""
    DESCRIPTION = 'ROIs reconstruction with brainrender'

    DEFAULT_ROI_COLORS = ['orange', 'magenta', 'dimgray']

    # ========== #
    # GROUP_ROIS #
    # ========== #

    GROUP_ROIS = 'ROI View Option'

    roi_region: str | tuple[str, ...] = argument(
        '--roi-region',
        metavar='NAME,...',
        type=str_tuple_type,
        default=(),
        group=GROUP_ROIS,
        help='only show rois in region(s)'
    )

    radius: float = argument(
        '--roi-radius',
        default=30,
        group=GROUP_ROIS,
        help='each roi radius'
    )

    roi_alpha: float = argument(
        '--roi-alpha',
        validator.float.in_range_closed(0, 1),
        default=1,
        group=GROUP_ROIS,
        help='region alpha value'
    )

    roi_colors: str | tuple[str, ...] = argument(
        '--roi-colors',
        metavar='COLOR,...',
        type=str_tuple_type,
        default=DEFAULT_ROI_COLORS,
        group=GROUP_ROIS,
        help='colors of rois per region'
    )

    region_col: str | None = argument(
        '--region-col',
        metavar='TREE..',
        default=None,
        group=GROUP_ROIS,
        help='if None, auto infer, and check the lowest merge level contain all the regions specified'
    )

    inverse_lut: bool = argument(
        '--inverse-lut',
        group=GROUP_ROIS,
        help='inverse right/left maps to ipsi/contra hemisphere look up table'
    )

    source_order: tuple[str, ...] | None = argument(
        '--source-order',
        metavar='SOURCE,...',
        type=str_tuple_type,
        default=None,
        group=GROUP_ROIS,
        help='source order to follow the roi_colors'
    )

    # ============== #
    # GROUP_ROI_LOAD #
    # ============== #

    GROUP_ROIS_LOAD = 'ROI load Option'

    classifier_file: Path | None = argument(
        '--classifier-file',
        validator.path.is_suffix('.csv').is_exists().optional(),
        metavar='FILE',
        type=Path,
        default=None,
        group=GROUP_ROIS_LOAD,
        help='csv output file from allenccf'
    )

    file: list[Path] | None = argument(
        '--file',
        validator.list().on_item(validator.path.is_suffix(['.csv', '.npy'])) | validator.optional(),
        metavar='FILE',
        type=list_type(Path),
        default=None,
        action='extend',
        group=GROUP_ROIS_LOAD,
        help="points file as 'npy' or 'csv'"
    )

    _need_close_file: list[NamedTemporaryFile] = []
    _point_file_list: list[str] = []

    def run(self):
        self.post_parsing()

        if not self._stop_render:
            self.render()
            self.render_output()

            if len(self._need_close_file) != 0:
                for f in self._need_close_file:
                    f.close()
                    Path(f.name).unlink(missing_ok=True)  # winOS

    def render(self):
        super().render()

        if self.classifier_file is not None:
            self._add_points_classifier_csv()

        if self.file is not None:
            self._add_points_generic_file()

        self._reconstruct_points_from_file()

    @property
    def _get_hemisphere_lut(self):
        if self.inverse_lut:
            return {'right': 'contra', 'left': 'ipsi', 'both': 'both'}
        else:
            return {'right': 'ipsi', 'left': 'contra', 'both': 'both'}

    def _add_points_classifier_csv(self):
        iter_coords = iter_source_coordinates(
            self.classifier_file,
            only_areas=self.roi_region,
            region_col=self.region_col,
            hemisphere=self._get_hemisphere_lut[self.hemisphere],
            to_brainrender=True if self.coordinate_space == 'ccf' else False,
            source_order=self.source_order
        )

        ret = [sc.coordinates for sc in iter_coords]
        self._save_tempfile(ret)

    def _save_tempfile(self, rois_list: list[list[np.ndarray]] | list[np.ndarray]):
        for p in rois_list:
            if isinstance(p, np.ndarray):
                # os handle for NamedTemporaryFile, https://stackoverflow.com/a/23212515
                delete = False if sys.platform == 'win32' else True
                f = NamedTemporaryFile(prefix='.temp-run-3d-proj-', suffix='.npy', delete=delete)

                np.save(f, p)
                f.seek(0)
                self._point_file_list.append(f.name)
                self._need_close_file.append(f)

    def _add_points_generic_file(self):
        for it in self.file:
            self._point_file_list.append(str(it))

    def _reconstruct_points_from_file(self, error_while_empty: bool = False):
        for i, file in enumerate(self._point_file_list):
            # type handle
            if file.endswith('.npy'):
                data = np.load(file)
            elif file.endswith('.csv'):
                data = pl.read_csv(file)
                data = as_coords_array(data)
                if self.coordinate_space == 'ccf':
                    data = allen_to_brainrender_coord(data)
            else:
                raise ValueError('Unsupported file type')

            # check
            if data.ndim != 2:
                raise ValueError(f'wrong dimension: {data.shape}')

            # add points
            if data.size == 0:
                if error_while_empty:
                    raise ValueError('no points found')
                else:
                    self.logger.warn('no points found')
            else:
                if data.shape[1] == 3:
                    colors = get_color(i, self.roi_colors)
                    self.logger.info(f'Plot Rois File: {i}, {file}, {colors}')
                    self.scene.add(Points(data, name='roi', colors=colors, alpha=self.roi_alpha, radius=self.radius, res=20))
                elif data.shape[1] == 4:  # TODO not test yet
                    k = data[:, 3].astype(int)
                    for t in np.unique(k):
                        self.scene.add(Points(
                            data[k == t, 0:3],
                            name='rois',
                            colors=get_color(t, self.roi_colors),
                            alpha=self.roi_alpha,
                            radius=self.radius
                        ))
                else:
                    raise ValueError(f'wrong shape: {data.shape}: {file}')


if __name__ == '__main__':
    RoiRenderCLI().main()
