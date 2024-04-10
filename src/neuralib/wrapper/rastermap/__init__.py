"""
RasterMap Wrapper
=================

.. seealso:: `<https://github.com/MouseLand/rastermap>`_
.. seealso:: `<https://colab.research.google.com/github/MouseLand/rastermap/blob/main/notebooks/rastermap_largescale.ipynb#scrollTo=E_WTZXx0Io0Y>`_



Example of 2P dataset pipeline
-----------------------


.. code-block:: python

    from __future__ import annotations

    import dataclasses

    import numpy as np
    import rastermap.utils
    from rastermap import Rastermap, rastermap

    from neuralib.plot import plot_figure, ax_merge
    from neuralib.util.util_type import PathLike
    from neuralib.wrapper.rastermap import RasterOptions, RasterMapResult, UserCluster


**1. Prepare container data input for rastermap**

    `Dimension parameters`:

        N = number of neurons

        T = number of image pulse

        S = number of stimulation (optional)


.. code-block:: python

    @dataclasses.dataclass
    class BaseRasterMapInput2P:
        xy_pos: np.ndarray  # soma central position (2, N)
        neural_activity: np.ndarray  # neural activity (N, T)
        image_time: np.ndarray  # 2p imaging time (T,)

        # optional user-specific behavioral variables
        position: np.ndarray | None
        velocity: np.ndarray | None
        pupil_area: np.ndarray | None
        stimulation_epoch: np.ndarray  # (S, 2) with on-off stim time

**2. Prepare config dict**

.. code-block:: python

    DEFAULT_2P_RASTERMAP_OPT: RasterOptions = {
        'n_clusters': 20,
        'n_PCs': 128,
        'locality': 0.75,
        'time_lag_window': 5,
        'grid_upsample': 10,
    }


**3. Run Rastermap and get results**

.. code-block:: python

    def run_rastermap_2p(dat: BaseRasterMapInput2P,
                         suite2p_directory: PathLike,
                         ops: RasterOptions | None = None,
                         neuron_bins: int = 30,
                         **kwargs) -> RasterMapResult:
        if ops is None:
            ops = DEFAULT_2P_RASTERMAP_OPT

        model = Rastermap(
            n_clusters=ops['n_clusters'],
            n_PCs=ops['n_PCs'],
            locality=ops['locality'],
            time_lag_window=ops['time_lag_window'],
            grid_upsample=ops['grid_upsample'],
            **kwargs
        ).fit(dat.neural_activity)

        embedding = model.embedding
        isort = model.isort

        sn = rastermap.utils.bin1d(dat.neural_activity[isort], bin_size=neuron_bins, axis=0)

        # For fit gui launch behavior
        _pseudo_cluster: UserCluster = {
            'ids': np.arange(10),
            'slice': slice(0, 10),
            'binsize': neuron_bins,
            'color': np.array([194.59459459, 255., 0., 50.])
        }

        ret = RasterMapResult(
            filename=str(suite2p_directory / 'F.npy'),
            save_path=str(suite2p_directory),
            isort=isort,
            embedding=embedding,
            ops=ops,
            user_clusters=[_pseudo_cluster],
            super_neurons=sn
        )

        return ret

**4. Plotting cluster and soma location color map**

.. code-block:: python

    def plot_rastermap_2p(raster: RasterMapResult, output: PathLike | None = None):
        with plot_figure(output, figsize=(6, 12)) as _ax:
            # heatmap
            ax = ax_merge(_ax)[:, :-1]
            ax.imshow(raster.super_neurons,
                      cmap="gray_r",
                      vmin=0,
                      vmax=0.8,
                      aspect="auto",
                      interpolation='none')
            ax.set(xlabel="time(s)", ylabel='superneurons')

            # soma colorbar
            ax = ax_merge(_ax)[:, -1:]
            ax.imshow(np.arange(0, raster.n_super)[:, np.newaxis], cmap="gist_ncar", aspect="auto")
            ax.axis("off")


.. code-block:: python

    def plot_rastermap_2p_soma(dat: BaseRasterMapInput2P,
                               raster: RasterMapResult,
                               output: PathLike | None = None):
        with plot_figure(output) as ax:
            ax.scatter(dat.xy_pos[0],
                       dat.xy_pos[1],
                       s=8, c=raster.embedding, cmap="gist_ncar", alpha=0.25)
            ax.invert_yaxis()
            ax.set(xlabel='X position(mm)', ylabel='Y position(mm)')
            ax.set_aspect('equal')





Example of Wfield imaging dataset pipeline
-----------------------


**1. Prepare container data input for rastermap**

    `Dimension parameters`:

        W = image width

        H = image height

        T = number of image pulse

        C = number of components after SVD reduction

        S = number of stimulation (optional)

.. code-block:: python


    from __future__ import annotations

    import dataclasses

    import numpy as np
    import rastermap.utils
    from rastermap import Rastermap
    from scipy.stats import zscore

    from neuralib.plot import plot_figure, ax_merge
    from neuralib.util.util_type import PathLike


    @dataclasses.dataclass
    class BaseRasterMapInputWField:
        sequences: np.ndarray  # image sequences (T, W, H)

        # singular vector
        n_components: int = dataclasses.field(init=False, default=128)
        sv: np.ndarray | None = dataclasses.field(init=False)  # (C, ) singular values
        Vsv: np.ndarray | None = dataclasses.field(init=False)  # (T, C) right singular vector
        U: np.ndarray | None = dataclasses.field(init=False)  # left singular vector (W * H, C)

        @property
        def n_frames(self) -> int:
            return self.sequences.shape[0]

        @property
        def width(self) -> int:
            return self.sequences.shape[1]

        @property
        def height(self) -> int:
            return self.sequences.shape[2]

        @property
        def xpos(self) -> np.ndarray:
            x = np.arange(self.width)
            y = np.arange(self.height)
            return np.meshgrid(x, y)[0]

        @property
        def ypos(self) -> np.ndarray:
            x = np.arange(self.width)
            y = np.arange(self.height)
            return np.meshgrid(x, y)[1]

        def compute_singular_vector(self) -> None:
            from sklearn.decomposition import TruncatedSVD

            # to 2D voxel
            seq = self.sequences.reshape(self.n_frames, self.height * self.width).T
            seq = seq - np.mean(seq, axis=1, keepdims=True)

            svd = TruncatedSVD(n_components=self.n_components)
            Vsv = svd.fit_transform(seq.T)
            U = seq @ (Vsv / (Vsv ** 2).sum(axis=0) ** 0.5)
            U /= (U ** 2).sum(axis=0) ** 0.5

            self.sv = svd.singular_values_
            self.Vsv = Vsv
            self.U = U


**2. Prepare config dict**

.. code-block:: python

    DEFAULT_WFIELD_RASTER_OPT: RasterOptions = {
        'n_clusters': 100,
        'locality': 0.5,
        'time_lag_window': 10,
        'grid_upsample': 10
    }


**3. Run Rastermap and get results**

.. code-block:: python


    def run_rastermap_2p(dat: BaseRasterMapInputWField,
                         ops: RasterOptions | None = None,
                         neuron_bins: int = 500,
                         **kwargs) -> RasterMapResult:
        if ops is None:
            ops = DEFAULT_WFIELD_RASTER_OPT

        ops['n_PCs'] = dat.n_components

        model = Rastermap(
            n_clusters=ops['n_clusters'],
            n_PCs=ops['n_PCs'],
            locality=ops['locality'],
            time_lag_window=ops['time_lag_window'],
            grid_upsample=ops['grid_upsample'],
            **kwargs
        ).fit(
            Usv=dat.U * dat.sv,  # left singular vectors weighted by the singular values
            Vsv=dat.Vsv  # right singular vectors weighted by the singular values
        )

        embedding = model.embedding
        isort = model.isort
        Vsv_sub = model.Vsv  # these are the PCs across time with the mean across voxels subtracted

        U_sn = rastermap.utils.bin1d(dat.U[isort], bin_size=neuron_bins, axis=0)  # bin over voxel axis
        sn = U_sn @ Vsv_sub.T
        sn = zscore(sn, axis=1)

        ret = RasterMapResult(
            filename=...,  # replace to user specific
            save_path=...,  # replace to user specific
            isort=isort,
            embedding=embedding,
            ops=ops,
            user_clusters=[],
            super_neurons=sn
        )

        return ret


**4. Plotting cluster and FOV color map**


.. code-block:: python

    def plot_rastermap_sort(raster: RasterMapResult,
                            output: PathLike | None = None):
        with plot_figure(output,
                         9, 20,
                         dpi=200,
                         gridspec_kw={'wspace': 1, 'hspace': 0.3}) as _ax:
            ax = ax_merge(_ax)[:, :-1]
            ax.imshow(raster.super_neurons,
                      cmap="gray_r",
                      vmin=0,
                      vmax=0.8,
                      aspect="auto",
                      interpolation='none')
            ax.set(xlabel="time(s)", ylabel='superneurons')

            # soma
            ax = ax_merge(_ax)[:, -1:]
            ax.imshow(np.arange(0, raster.n_super)[:, np.newaxis], cmap="gist_ncar", aspect="auto")
            ax.axis("off")


    def plot_raster_voxel(dat: BaseRasterMapInputWField,
                          raster: RasterMapResult,
                          output: PathLike | None = None):
        with plot_figure(output) as ax:
            ax.scatter(dat.xpos,
                       dat.ypos,
                       s=1, c=raster.embedding, cmap="gist_ncar", alpha=0.25)
            ax.invert_yaxis()
            ax.set(xlabel='X position (um)', ylabel='Y position')
            ax.set_aspect('equal')


"""

from .core import *
