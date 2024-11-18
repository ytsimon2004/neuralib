from __future__ import annotations

from pathlib import Path
from typing import Union, Iterator, TYPE_CHECKING, Optional, Final

import numpy as np

from neuralib.ephys.glx import EphysRecording
from .cluster_info import ClusterInfo
from .params import KilosortParameter

if TYPE_CHECKING:
    from .ops_ks4 import Kilosort4Options
    from .result import KilosortResult

__all__ = ['KilosortFiles']


class KilosortFiles:
    """
    A directory that stored the results from Kilosort and Phy.
    """

    def __init__(self, directory: Union[str, Path], ephys: EphysRecording | None = None):
        """

        :param directory: storage directory.
        :param ephys: source ephys data.
        """
        self.directory = Path(directory)
        self.ephys = ephys

    def __str__(self):
        return f'KilosortFiles(directory={self.directory})'

    def __repr__(self):
        return str(self)

    # ======== #
    # get file #
    # ======== #

    def get_file(self, name: str, create=False) -> Path:
        """
        Return found file with *name*.

        :param name: filename
        :param create: Ignored. Used by `KilosortProcessedFiles`.
        :return: filepath
        """
        return self.directory / name

    def list_cluster_data_name(self) -> list[str]:
        """Find all cluster_NAME.tsv files.

        :return: list of NAME
        """
        return [
            it.stem.replace('cluster_', '')
            for it in self.directory.glob('cluster_*.tsv')
        ]

    def get_cluster_data_file(self, name: str) -> Path:
        return self.get_file(f'cluster_{name}.tsv')

    def glob(self, glob: str, *, recursive=True, unique=True) -> Iterator[Path]:
        """
        Find files by a *glob* pattern.

        :param glob:
        :param recursive: Ignored. Used by `KilosortProcessedFiles`.
        :param unique: Ignored. Used by `KilosortProcessedFiles`.
        :return:
        """
        return self.directory.glob(glob)

    # ========= #
    # filepaths #
    # ========= #

    @property
    def parameter_file(self) -> Path:
        return self.get_file('params.py')

    @property
    def cluster_info_file(self) -> Path:
        return self.get_file('cluster_info.tsv')

    @property
    def cluster_contam_file(self) -> Path:
        return self.get_cluster_data_file('ContamPct')

    @property
    def cluster_amplitude_file(self) -> Path:
        return self.get_cluster_data_file('Amplitude')

    @property
    def cluster_group_file(self) -> Path:
        return self.get_file('cluster_group.tsv')

    @property
    def channel_map_file(self) -> Path:
        """the channel map, i.e. which row of the data file to look in for the channel in question """
        return self.get_file('channel_map.npy')

    @property
    def channel_pos_file(self) -> Path:
        """matrix with each row giving the x and y coordinates of that channel.

        Together with the channel map, this determines how waveforms will be plotted in WaveformView
        """
        return self.get_file('channel_positions.npy')

    @property
    def spike_time_file(self) -> Path:
        return self.get_file('spike_times.npy')

    @property
    def spike_cluster_file(self) -> Path:
        """giving the cluster identity of each spike."""
        return self.get_file('spike_clusters.npy')

    @property
    def spike_amplitude_file(self) -> Path:
        """the amplitude scaling factor that was applied to the template when extracting that spike"""
        return self.get_file('amplitudes.npy')

    @property
    def spike_template_file(self) -> Path:
        """specifying the identity of the template that was used to extract each spike"""
        return self.get_file('spike_templates.npy')

    @property
    def template_file(self) -> Path:
        """giving the template shapes on the channels given in templates_ind.npy"""
        return self.get_file('templates.npy')

    @property
    def whitening_mat_file(self) -> Path:
        """matrix applied to the data during automatic spike sorting"""
        return self.get_file('whitening_mat.npy')

    @property
    def whitening_invmat_file(self) -> Path:
        """inverse of the whitening matrix."""
        return self.get_file('whitening_mat_inv.npy')

    @property
    def feature_file(self) -> Path:
        """matrix giving the PC values for each spike.

        This file doesn't generate when kilosort's templateFeatures is empty
        """
        return self.get_file('pc_features.npy')

    @property
    def feature_index_file(self) -> Path:
        """matrix specifying which pcFeatures are included in the pc_features matrix.

        This file doesn't generate when kilosort's templateFeatures is empty
        """
        return self.get_file('pc_features_ind.npy')

    @property
    def similar_templates_file(self) -> Path:
        """matrix giving the similarity score (larger is more similar) between each pair of templates"""
        return self.get_file('similar_templates.npy')  # shape (T, T)

    # templates_ind.npy - (cluster, channels) matrix
    #   specifying the channels on which each template is defined.
    #   In the case of Kilosort templates_ind is just the integers from 0 to nChannels-1,
    #   since templates are defined on all channels.

    # template_feature_ind.npy - (T, F) u32 matrix
    #   specifying which templateFeatures are included in the template_features matrix.

    # (kilosort4) kept_spikes.npy
    # (kilosort4) spike_detection_templates.npy
    # (kilosort4) whitening_mat_dat.npy

    @property
    def motion_file(self) -> Path:
        """
        kilosort3 specific file.
        """
        if not (p := self.get_file('motion.npy')).exists():
            raise FileNotFoundError(p)
        return p

    @property
    def options_file(self) -> Path:
        """
        kilosort4 specific file.
        """
        return self.get_file('ops.npy')

    # ==== #
    # data #
    # ==== #

    @property
    def total_channels(self) -> int:
        """n_channels_dat"""
        return KilosortParameter.get_total_channels(self.parameter_file)

    @property
    def recording_data_file(self) -> Path:
        """dat_path"""
        return KilosortParameter.get_data_path(self.parameter_file)

    def parameter_data(self) -> KilosortParameter:
        """param.py"""
        return KilosortParameter.read(self.parameter_file)

    def options_data(self) -> Kilosort4Options:
        """ops.npy"""
        if not (file := self.options_file).exists():
            raise FileNotFoundError(file)
        return np.load(file).item()

    def cluster_info(self, *, raw=False) -> ClusterInfo:
        """
        Read cluster info from `cluster_info.tsv`.

        :param raw: return dataframe without any fixing.
        :return:
        :raises FileNotFoundError: `cluster_info.tsv` is not yet created (usually by Phy).
        :seealso: fix_cluster_info
        """
        ret = ClusterInfo.read_csv(self.cluster_info_file)
        if raw:
            return ret

        from .cluster_info import fix_cluster_info
        return fix_cluster_info(ret, self.ephys, self.result())

    def cluster_data(self, name: str) -> ClusterInfo:
        """cluster_NAME.tsv"""
        return ClusterInfo.read_csv(self.get_cluster_data_file(name))

    # ====== #
    # result #
    # ====== #

    def result(self) -> KilosortResult:
        from .result import KilosortResult
        return KilosortResult(self)


class KilosortProcessedFiles(KilosortFiles):
    """
    A directory that stored the post-processing results.

    We do not put the post-processing results back to the original kilosort result directory,
    which is considered secondary **raw data**. This class keep the KilosortFiles/KilosortResult
    working in the same way, but just separated the raw kilosort results and the processed results.

    * Benefits

        * keep the original results, so changing the processing method (while testing) is much easily.

    * Shortage

        * TBD

    """

    KS_SRC_MARK: Final[str] = '.ks_src'

    def __init__(self, directory: Path, ks_file: KilosortFiles = None, processed: Optional[bool] = None):
        """

        :param directory: post-processing directory
        :param ks_file: source kilosort results. It could be another `KilosortProcessedFiles`,
            but make sure use the same directory in after using. Otherwise, an error wii be raised.
            It can be auto-instanced by reading `.ks_src` marker file.
        :param processed: policy of finding the files.
        :raises RuntimeError: First time create a post-processing directory without giving the source *ks_file*.
            Or the giving *ks_file* does not match to the previous used source *ks_file*.
        """
        ks_file = self._load_ks_file(directory, ks_file)
        super().__init__(directory, ks_file.ephys)
        self.ks_file = ks_file
        self.__processed = processed

    @classmethod
    def _load_ks_file(cls, directory: Path, ks_file: KilosortFiles = None) -> KilosortFiles:
        ks_src = directory / cls.KS_SRC_MARK
        if not ks_src.exists():
            if ks_file is None:
                raise RuntimeError('missing source KilosortFiles')
            else:
                # write marker file
                with ks_src.open('w') as f:
                    print(ks_file.directory.absolute(), file=f)
        else:
            # read marker file
            ks_dir = Path(ks_src.read_text())
            if ks_file is None:  # auto-instanced KilosortFiles or KilosortProcessedFiles
                if (ks_dir / cls.KS_SRC_MARK).exists():
                    ks_file = KilosortProcessedFiles(ks_dir)
                else:
                    ks_file = KilosortFiles(ks_dir)
            else:  # check source
                if ks_file.directory.absolute() != ks_dir:
                    raise RuntimeError(f'{directory} does not derivative from {ks_file}')

        return ks_file

    def with_processed(self, processed: Optional[bool]) -> KilosortProcessedFiles:
        """
        change the *processed* policy.

        * `None` return the filepath under the post-processing directory if it is existed.
            Otherwise, return the file path from the source directory.
        * `True` always return the file path from the post-processing directory.
        * `False` always return the file path from the source directory.

        Use case:

        * when reading the data: use *processed* `None` policy
        * when post-processing: use *processed* `False` policy to read, and `True` to write.
            Or using `get_file(create=True)` to get a creating-purposed a filepath under the post-processing directory.

        :param processed: behavior of finding file.
        :return:
        """
        if processed == self.__processed:
            return self
        return KilosortProcessedFiles(self.directory, self.ks_file, processed)

    def get_file(self, name: str, create=False) -> Path:
        """
        Return found file with *name*.

        :param name: filename
        :param create: the file is used to be created, so pass the exist checking.
        :return: filepath
        """
        p = self.directory / name
        if self.__processed is True or create:
            return p

        if self.__processed is None and p.exists():
            return p

        return self.ks_file.get_file(name)

    def glob(self, glob: str, *, recursive=True, unique=True) -> Iterator[Path]:
        """
        Find files by a *glob* pattern.

        :param glob:
        :param recursive: Also find files from source.
        :param unique: unique on filename.
        :return:
        """
        current = self.directory.glob(glob)
        if not recursive:
            return current

        parent = self.ks_file.glob(glob, recursive=recursive, unique=unique)

        if unique:
            results = {p.name: p for p in parent}
            for p in current:
                results[p.name] = p  # may replace parent's result
            ret = iter(results.values())

        else:
            results = list(parent)
            results.extend(list(current))
            ret = iter(results)

        return ret


def shadow_ks_directory(ks_file: KilosortFiles,
                        directory: Path,
                        copy_files: list[str] = None,
                        ignore_files: list[str] = None,
                        rename_files: dict[str, str] = None,
                        force: bool = False,
                        overwrite: bool = False) -> KilosortProcessedFiles:
    """
    Prepare a shadowed kilosort result directory with some file modification.

    :param ks_file: source kilosort result
    :param directory: shadow directory
    :param copy_files: copy files instead of making a link
    :param ignore_files: ignore files
    :param rename_files: rename files. A dictionary with `{old_name: new_name}`.
        Make sure *copy_files* and *ignore_files* use the *new_name* instead of the *old_name*.
    :param force: force create link even if the source file is not existed.
    :param overwrite: overwrite files in *copy_files*. Link files are always relinked.
    :return: The shadowed kilosort result directory.
    :seealso: shadow_phy_directory
    :seealso: shadow_si_directory
    """
    import shutil
    copy_files = copy_files or []
    ignore_files = ignore_files or []
    rename_files = rename_files or {}

    directory.mkdir(parents=True, exist_ok=True)

    # cluster_info.tsv

    filepath = directory / 'cluster_info.tsv'
    if not filepath.exists() or overwrite:
        shutil.copyfile(ks_file.cluster_info_file, filepath)

    # symlink other files

    filename_list = [
        "amplitudes.npy",
        "channel_map.npy",
        "channel_positions.npy",
        "pc_feature_ind.npy",
        "pc_features.npy",
        "similar_templates.npy",
        "spike_clusters.npy",
        "spikeinterface_log.json",
        "spikeinterface_params.json",
        "spike_templates.npy",
        "spike_times.npy",
        "template_feature_ind.npy",
        "template_features.npy",
        "templates_ind.npy",
        "templates.npy",
        "whitening_mat_inv.npy",
        "whitening_mat.npy",
        "recording.dat"  # TODO Is it always named as recording.dat?
        # ks_file.parameter_data().data_path.name
    ]

    for cluster_data_file in ks_file.glob('cluster_*.tsv', recursive=True, unique=True):
        # 'cluster_KSLabel.tsv',
        # 'cluster_group.tsv',
        filename_list.append(cluster_data_file.name)

    # params.py
    param = KilosortParameter.read(ks_file.parameter_file)
    params_path = directory / 'params.py'
    params_path.unlink(missing_ok=True)  # avoid write content on link

    with params_path.open('w') as f:
        print('dat_path', '=', '"recording.dat"', file=f)
        print('n_channels_dat', '=', param.channel_number, file=f)
        print('dtype', '=', repr(param.data_type), file=f)
        print('offset', '=', param.channel_offset, file=f)
        print('sample_rate', '=', param.sample_rate, file=f)
        print('hp_filtered', '=', param.hp_filtered, file=f)

    # copy/link/ignore files

    for filename in filename_list:
        filepath = directory / rename_files.get(filename, filename)

        if filename in ignore_files:
            pass
        elif not filepath.exists() or overwrite:
            if filename in copy_files:
                shutil.copyfile(ks_file.get_file(filename), filepath)
            else:
                filepath.unlink(missing_ok=True)
                src = ks_file.get_file(filename).absolute()
                if src.exists() or force:
                    filepath.symlink_to(src)

    return KilosortProcessedFiles(directory, ks_file)


def shadow_phy_directory(ks_file: KilosortFiles,
                         tmp_directory: Path,
                         overwrite: bool = False) -> KilosortProcessedFiles:
    """
    Create a shadow directory for phy.

    This function is used to create a temporary directory to save the phy's result, and
    prevent from phy to modify `cluster_group.tsv` and `spike_clusters.npy` files.

    Remember to copy the files mentioned in the above back after phy curation.

    :param ks_file: Which kilosort result directory need to be shadowed
    :param tmp_directory: shadow directory
    :param overwrite: overwrite the shadow directory if it is existed.
    :return:
    """
    return shadow_ks_directory(ks_file, tmp_directory, overwrite=overwrite, copy_files=[
        'cluster_group.tsv',
        'spike_clusters.npy',
    ], ignore_files=[
        "spikeinterface_log.json",
        "spikeinterface_params.json",
    ])


def shadow_si_directory(ks_file: KilosortFiles,
                        tmp_directory: Path,
                        overwrite: bool = False) -> KilosortProcessedFiles:
    """
    Create a shadow directory for spikeinterface.

    :param ks_file: Which kilosort result directory need to be shadowed
    :param tmp_directory: shadow directory
    :param overwrite: overwrite the shadow directory if it is existed.
    :return:
    """
    return shadow_ks_directory(ks_file, tmp_directory, overwrite=overwrite, rename_files={
        'spikeinterface_log.json': 'log.json',
        'spikeinterface_params.json': 'params.json',  # TODO does spikeinterface_params.json is a file needed by SI?
    })
