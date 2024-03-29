from typing import Optional

import numpy as np

from neuralib.argp import argument, str_tuple_type
from neuralib.atlas.brainrender.core import BrainReconstructor, RoiType
from rscvp.util.util_ccf import iter_source_coordinates


class RoisReconstructor(BrainReconstructor):
    DESCRIPTION = 'For labelled rois reconstruction from 2dccf pipeline'

    only_roi_region: str = argument(
        '--roi-region',
        metavar='NAME,...',
        type=str_tuple_type,
        default=[],
        help='only show rois in region(s)'
    )

    region_col: Optional[str] = argument(
        '--region-col',
        metavar='MERGE_AC..',
        default=None,
        help='if None, auto infer, and check the lowest merge level contain all the regions specified'
    )

    def load_merge_points(self, rois: np.ndarray = None) -> RoiType:
        """
        load from 2dccf pipeline, with different channels of fluorescence labelled points

        :param rois: (N', 3) with xyz(ap, dv, ml). i.e., prelocate roi, reference points ...
        :return:
        """
        ret = []
        if rois is not None:
            if rois.shape[1] != 3:
                raise ValueError('')
            ret.append(rois)

        if self.csv_file is not None:
            iter_coords = iter_source_coordinates(self.csv_file,
                                                  only_areas=self.only_roi_region,
                                                  region_col=self.region_col)
            for sc in iter_coords:
                ret.append(sc.coordinates)

        return ret

    def load(self):
        if self.csv_file is not None:
            rois_list = self.load_merge_points()
            self.add_points(rois_list)


if __name__ == '__main__':
    RoisReconstructor().main()
