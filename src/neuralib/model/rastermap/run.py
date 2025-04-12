import numpy as np

from .core import RasterOptions, UserCluster, RasterMapResult


def run_rastermap(neural_activity: np.ndarray,
                  bin_size: int, *,
                  options: RasterOptions | None = None,
                  filename: str | None = None,
                  save_path: str | None = None) -> RasterMapResult:
    """
    Run rastermap

    :param neural_activity: neural activity. `Array[float, [N, T]]`
    :param bin_size: number of neurons to bin over
    :param options: :class:`~neuralib.model.rastermap.core.RasterOptions`
    :param filename: optional specify if GUI re-launch
    :param save_path: optional specify if GUI re-launch
    """
    rastermap = RunRastermap(neural_activity, bin_size, options, filename, save_path)
    return rastermap.run()


class RunRastermap:
    DEFAULT_OPTIONS: RasterOptions = {
        'n_clusters': 100,
        'n_PCs': 128,
        'locality': 0.75,
        'time_lag_window': 5,
        'grid_upsample': 10,
    }

    def __init__(self, neural_activity: np.ndarray,
                 bin_size: int,
                 options: RasterOptions | None = None,
                 filename: str | None = None,
                 save_path: str | None = None):
        """

        :param neural_activity: neural activity. `Array[float, [N, T]]`
        :param bin_size: number of neurons to bin over
        :param options: :class:`~neuralib.model.rastermap.core.RasterOptions`
        :param filename: optional specify if GUI re-launch
        :param save_path: optional specify if GUI re-launch
        """

        self.neural_activity = neural_activity
        self.bin_size = bin_size

        self.filename = filename
        self.save_path = save_path

        self._options = options or self.DEFAULT_OPTIONS

    @property
    def options(self) -> RasterOptions:
        return self._options

    @property
    def _user_cluster(self) -> UserCluster:
        """pseudo, for fit gui launch behavior"""
        return {
            'ids': np.arange(10),
            'slice': slice(0, 10),
            'binsize': self.bin_size,
            'color': np.array([194.59459459, 255., 0., 50.])
        }

    def run(self, **kwargs) -> RasterMapResult:
        try:
            from rastermap import Rastermap
            import rastermap.utils
        except ImportError:
            raise RuntimeError('pip install rastermap first')

        opt = self.options
        model = Rastermap(n_clusters=opt['n_clusters'],
                          n_PCs=opt['n_PCs'],
                          locality=opt['locality'],
                          time_lag_window=opt['time_lag_window'],
                          grid_upsample=opt['grid_upsample'],
                          **kwargs)

        model.fit(self.neural_activity)

        embedding = model.embedding
        isort = model.isort

        sn = rastermap.utils.bin1d(self.neural_activity[isort], bin_size=self.bin_size, axis=0)

        ret = RasterMapResult(
            filename=self.filename,
            save_path=self.save_path,
            isort=isort,
            embedding=embedding,
            ops=self.options,
            user_clusters=[self._user_cluster],
            super_neurons=sn
        )

        if self.save_path is not None:
            ret.save(self.save_path)

        return ret
