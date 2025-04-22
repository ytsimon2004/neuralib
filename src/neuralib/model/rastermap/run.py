import pickle
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.stats import zscore

from neuralib.typing import PathLike
from .core import RasterOptions, RasterMapResult

__all__ = ['DATA_TYPE', 'run_rastermap']

DATA_TYPE = Literal['cellular', 'wfield']


def run_rastermap(neural_activity: np.ndarray,
                  bin_size: int,
                  dtype: DATA_TYPE, *,
                  options: RasterOptions | None = None,
                  svd_components: int = 128,
                  svd_cache: PathLike | None = None,
                  invalid_svd_cache: bool = False,
                  filename: str | None = None,
                  save_path: str | None = None,
                  **kwargs) -> RasterMapResult:
    """
    Run rastermap

    :param neural_activity: neural activity. `Array[float, [N, T]]` | `Array[Any, [T, H, W]]`
    :param bin_size: number of neurons to bin over
    :param dtype: :attr:`~neuralib.model.rastermap.run.DATA_TYPE`
    :param options: :class:`~neuralib.model.rastermap.core.RasterOptions`
    :param svd_components: number of SVD components to use (for widefield dtype)
    :param svd_cache: path to cache SVD file (for widefield dtype)
    :param invalid_svd_cache: invalid (recompute) SVD cached file (for widefield dtype)
    :param filename: optional specify if GUI re-launch
    :param save_path: optional specify if GUI re-launch
    """
    rastermap = RunRastermap(neural_activity, bin_size, dtype,
                             options=options,
                             svd_components=svd_components,
                             svd_cache=svd_cache,
                             invalid_svd_cache=invalid_svd_cache,
                             filename=filename,
                             save_path=save_path)
    return rastermap.run(**kwargs)


class RunRastermap:
    DEFAULT_OPTIONS: RasterOptions = {
        'n_clusters': 50,
        'n_PCs': 128,
        'locality': 0.75,
        'time_lag_window': 5,
        'grid_upsample': 10,
    }

    def __init__(self, neural_activity: np.ndarray,
                 bin_size: int,
                 dtype: DATA_TYPE, *,
                 options: RasterOptions | None = None,
                 svd_components: int = 128,
                 svd_cache: PathLike | None = None,
                 invalid_svd_cache: bool = False,
                 filename: str | None = None,
                 save_path: str | None = None):
        """

        :param neural_activity: neural activity. `Array[float, [N, T]]` | `Array[Any, [T, H, W]]`
        :param bin_size: number of neurons to bin over
        :param options: :class:`~neuralib.model.rastermap.core.RasterOptions`
        :param svd_components: number of SVD components to use (for widefield dtype)
        :param svd_cache: path to cache SVD file (for widefield dtype)
        :param invalid_svd_cache: invalid (recompute) SVD cached file (for widefield dtype)
        :param filename: optional specify if GUI re-launch
        :param save_path: optional specify if GUI re-launch
        """

        self.neural_activity = neural_activity
        self.bin_size = bin_size
        self.dtype = dtype

        self.svd_components = svd_components
        self.svd_cache = svd_cache
        self.invalid_svd_cache = invalid_svd_cache

        self.filename = filename
        self.save_path = save_path

        self._options = options or self.DEFAULT_OPTIONS

    @property
    def options(self) -> RasterOptions:
        return self._options

    def run(self, **kwargs) -> RasterMapResult:
        """run rastermap with the given dtype"""
        match self.dtype:
            case 'cellular':
                ret = self.run_cellular(**kwargs)
            case 'wfield':
                ret = self.run_wfield(**kwargs)
            case _:
                raise NotImplementedError('')

        if self.save_path is not None:
            ret.save(self.save_path)

        return ret

    def run_cellular(self, **kwargs) -> RasterMapResult:
        """cellular input run. `Array[float, [N, T]]`"""
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
            ops=opt,
            user_clusters=[],
            super_neurons=sn
        )

        return ret

    def run_wfield(self, **kwargs):
        """wfield input run. `Array[Any, [T, H, W]]`"""
        from neuralib.imaging.widefield import compute_singular_vector

        try:
            from rastermap import Rastermap
            import rastermap.utils
        except ImportError:
            raise RuntimeError('pip install rastermap first')

        #
        opt = self.options
        model = Rastermap(
            n_clusters=opt['n_clusters'],
            n_PCs=opt['n_PCs'],
            locality=opt['locality'],
            time_lag_window=opt['time_lag_window'],
            grid_upsample=opt['grid_upsample'],
            **kwargs
        )

        # svd cache
        if self.svd_cache is not None:
            file = Path(self.svd_cache)
            if file.exists() and not self.invalid_svd_cache:
                with file.open("rb") as f:
                    sv = pickle.load(f)
            else:
                sv = compute_singular_vector(self.neural_activity, self.svd_components)
                with file.open('wb') as f:
                    pickle.dump(sv, f)
        else:
            sv = compute_singular_vector(self.neural_activity, self.svd_components)

        usv = sv.singular_value * sv.left_vector
        vsv = sv.right_vector

        # fit
        model.fit(Usv=usv, Vsv=vsv)
        embedding = model.embedding
        isort = model.isort
        Vsv_sub = model.Vsv

        U_sn = rastermap.utils.bin1d(sv.left_vector[isort], bin_size=self.bin_size, axis=0)
        sn = U_sn @ Vsv_sub.T
        sn = zscore(sn, axis=1)

        ret = RasterMapResult(
            filename=self.filename,
            save_path=self.save_path,
            isort=isort,
            embedding=embedding,
            ops=opt,
            user_clusters=[],
            super_neurons=sn
        )

        return ret
