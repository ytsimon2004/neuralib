import shutil
from pathlib import Path
from typing import TypedDict, Final, Literal, get_args

import attrs
import polars as pl

from neuralib.atlas.ccf.core import (
    AbstractCCFDir,
    SagittalCCFDir,
    SagittalCCFOverlapDir,
    CoronalCCFOverlapDir,
    CoronalCCFDir
)
from neuralib.atlas.ccf.norm import ROIS_NORM_TYPE
from neuralib.atlas.typing import Area, HEMISPHERE_TYPE, Source, Channel
from neuralib.atlas.util import PLANE_TYPE
from neuralib.util.logging import setup_clogger, LOGGING_IO_LEVEL
from neuralib.util.utils import uglob

# TODO to class level?
Logger = setup_clogger(caller_name=Path(__file__).name)

FluorReprType = dict[Channel, Source]


# ======= #
# Configs #
# ======= #

class UserInjectionConfig(TypedDict):
    area: Area
    """injection area"""
    hemisphere: HEMISPHERE_TYPE
    """injection hemisphere"""
    ignore: bool
    """whether local roi counts will be ignored"""
    fluor_repr: FluorReprType
    """fluorescence color and tracing source alias pairs"""


# Example (replace to user-specific) TODO as doc
_DEFAULT_RSP_CONFIG = UserInjectionConfig(
    area='RSP',
    hemisphere='ipsi',
    ignore=True,
    fluor_repr=dict(
        rfp='pRSC',
        gfp='aRSC',
        overlap='overlap'
    )
)

# ========== #
# Classifier #
# ========== #

CLASSIFIED_COL = Literal[
    'acronym_abbr',
    'merge_ac_0',
    'merge_ac_1',
    'merge_ac_2',
    'merge_ac_3',
    'merge_ac_4',
    'family'
]


class RoiClassifier:
    __slots__ = ('ccf_dir', 'merge_level', 'plane',
                 'ignore_injection_site', 'fluor_repr', '_fluor_order',
                 '_injection_area', '_injection_hemi', '_parse_df')

    def __init__(self, ccf_dir: AbstractCCFDir,
                 merge_level: int | str | None = None,
                 plane: PLANE_TYPE = 'coronal',
                 config: UserInjectionConfig | None = None):
        """
        Classifier for see the ROIs distribution across the mouse brain

        :param ccf_dir: ccf directory info
        :param merge_level: which number (int) of hierarchical merge area, or the col name (str) for output generate
        :param plane: slice cutting orientation
        :param config: `UserInjectionConfig` for customized injection info
        """

        self.ccf_dir = ccf_dir
        self.merge_level = merge_level
        self.plane = plane

        # config
        if config is None:
            config = _DEFAULT_RSP_CONFIG

        self.ignore_injection_site: Final[bool] = config['ignore']
        self.fluor_repr: FluorReprType = config['fluor_repr']  # constructs can be swapped if experimental need

        self._fluor_order: Final[tuple[Channel, ...]] = tuple(list(config['fluor_repr'].keys()))
        self._injection_area: Final[Area] = config['area']  # use startswith to ignore
        self._injection_hemi: Final[HEMISPHERE_TYPE] = config['hemisphere']

        # cache
        self._parse_df: pl.DataFrame | None = None

    @property
    def has_overlap(self) -> bool:
        """If has overlap channel counts"""
        config_check = 'overlap' in self.fluor_repr
        if self.ccf_dir.with_overlap_sources != config_check:
            raise RuntimeError('check UserInjectionConfig and AbstractCCFDir')
        return config_check

    @property
    def parsed_df(self) -> pl.DataFrame:
        """dataframe after parsing

        Example::

            ┌───────────────────────────────────┬─────────┬─────────────┬─────────────┬─────────────┬─────────┬─────────┬────────┬───────────────────────────┬──────────────┬────────┬────────────┬────────────┬────────────┬────────────┬────────────┬───────────┐
            │ name                              ┆ acronym ┆ AP_location ┆ DV_location ┆ ML_location ┆ avIndex ┆ channel ┆ source ┆ abbr                      ┆ acronym_abbr ┆ hemi.  ┆ merge_ac_0 ┆ merge_ac_1 ┆ merge_ac_2 ┆ merge_ac_3 ┆ merge_ac_4 ┆ family    │
            │ ---                               ┆ ---     ┆ ---         ┆ ---         ┆ ---         ┆ ---     ┆ ---     ┆ ---    ┆ ---                       ┆ ---          ┆ ---    ┆ ---        ┆ ---        ┆ ---        ┆ ---        ┆ ---        ┆ ---       │
            │ str                               ┆ str     ┆ f64         ┆ f64         ┆ f64         ┆ i64     ┆ str     ┆ str    ┆ str                       ┆ str          ┆ str    ┆ str        ┆ str        ┆ str        ┆ str        ┆ str        ┆ str       │
            ╞═══════════════════════════════════╪═════════╪═════════════╪═════════════╪═════════════╪═════════╪═════════╪════════╪═══════════════════════════╪══════════════╪════════╪════════════╪════════════╪════════════╪════════════╪════════════╪═══════════╡
            │ Ectorhinal area/Layer 5           ┆ ECT5    ┆ -3.03       ┆ 4.34        ┆ -4.5        ┆ 377     ┆ gfp     ┆ aRSC   ┆ Ectorhinal area           ┆ ECT          ┆ contra ┆ ECT        ┆ ECT        ┆ ECT        ┆ ECT        ┆ ECT        ┆ ISOCORTEX │
            │ Perirhinal area layer 6a          ┆ PERI6a  ┆ -3.03       ┆ 4.42        ┆ -4.37       ┆ 372     ┆ gfp     ┆ aRSC   ┆ Perirhinal area           ┆ PERI         ┆ contra ┆ PERI       ┆ PERI       ┆ PERI       ┆ PERI       ┆ PERI       ┆ ISOCORTEX │
            │ …                                 ┆ …       ┆ …           ┆ …           ┆ …           ┆ …       ┆ …       ┆ …      ┆ …                         ┆ …            ┆ …      ┆ …          ┆ …          ┆ …          ┆ …          ┆ …          ┆ …         │
            │ Ventral auditory area layer 6a    ┆ AUDv6a  ┆ -2.91       ┆ 3.52        ┆ 4.46        ┆ 156     ┆ rfp     ┆ pRSC   ┆ Ventral auditory area     ┆ AUDv         ┆ ipsi   ┆ AUD        ┆ AUD        ┆ AUD        ┆ AUD        ┆ AUDv       ┆ ISOCORTEX │
            │ Ectorhinal area/Layer 6a          ┆ ECT6a   ┆ -2.91       ┆ 4.14        ┆ 4.47        ┆ 378     ┆ rfp     ┆ pRSC   ┆ Ectorhinal area           ┆ ECT          ┆ ipsi   ┆ ECT        ┆ ECT        ┆ ECT        ┆ ECT        ┆ ECT        ┆ ISOCORTEX │
            │ Temporal association areas layer… ┆ TEa5    ┆ -2.91       ┆ 4.02        ┆ 4.55        ┆ 365     ┆ rfp     ┆ pRSC   ┆ Temporal association area ┆ TEa          ┆ ipsi   ┆ TEa        ┆ TEa        ┆ TEa        ┆ TEa        ┆ TEa        ┆ ISOCORTEX │
            └───────────────────────────────────┴─────────┴─────────────┴─────────────┴─────────────┴─────────┴─────────┴────────┴───────────────────────────┴──────────────┴────────┴────────────┴────────────┴────────────┴────────────┴────────────┴───────────┘
        """
        if self._parse_df is None:
            df = self._cache_parsed_dataframe()

            if self.ignore_injection_site:
                df = self._ignore_injection_site(df)
            self._parse_df = df

        return self._parse_df

    @parsed_df.setter
    def parsed_df(self, df: pl.DataFrame):
        """For supply overlap purpose"""
        self._parse_df = df

    # ============== #
    # Pre-Processing #
    # ============== #
    #
    # def _cache_parsed_dataframe(self, force_save: bool = False) -> pl.DataFrame:
    #     """concat & parse & add fields and cache a parsed dataframe"""
    #     file = self.ccf_dir.parse_csv
    #
    #     if not file.exists() or force_save:
    #         df = parse_csv(self.ccf_dir, self.fluor_repr, plane=self.plane)
    #
    #         ac = df['acronym']
    #
    #         # add merge level cols
    #         df = df.with_columns(
    #             pl.Series(name=f'merge_ac_{level}', values=merge_until_level(ac, level))
    #             for level in range(NUM_MERGE_LAYER)
    #         )
    #
    #         # add family col
    #         def categorize_family(row) -> str:
    #             for name, family in DEFAULT_FAMILY_DICT.items():
    #                 if row in family:
    #                     return name
    #
    #         df = df.with_columns(pl.col('merge_ac_0')
    #                              .map_elements(categorize_family, return_dtype=pl.Utf8)
    #                              .fill_null('unknown')
    #                              .alias('family'))
    #
    #         df.write_csv(file)
    #     else:
    #         df = pl.read_csv(file)

    # return df

    def _ignore_injection_site(self, df: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
        area = self._injection_area
        hemi = self._injection_hemi
        Logger.info(f'remove {area} in {hemi} in {self.ccf_dir.animal}')

        expr1 = pl.col('acronym').str.starts_with(area)
        expr2 = pl.col('hemi.') == hemi

        if verbose:
            exc = (
                df.filter(pl.col('acronym').str.starts_with(area))
                .group_by('hemi.').agg(pl.len())
            )

            print(exc)

        return df.filter(~(expr1 & expr2))

    # ===== #
    # Utils #
    # ===== #

    # # noinspection PyTypeChecker
    # @property
    # def classified_column(self) -> CLASSIFIED_COL:
    #     """ """
    #     if self.merge_level is None:
    #         return 'acronym_abbr'
    #
    #     if isinstance(self.merge_level, int):
    #         return f'merge_ac_{self.merge_level}'
    #     elif isinstance(self.merge_level, str) and self.merge_level in self.parsed_df.columns:
    #         return self.merge_level
    #     else:
    #         raise ValueError(f'invalid merge level: {self.merge_level}')

    # @property
    # def channels(self) -> list[Channel]:
    #     """list of fluorescence channels
    #     (sorted based on :class:`~neuralib.atlas.ccf.classifier.UserInjectionConfig.fluor_repr`)"""
    #     chan_list = self.parsed_df['channel'].unique()
    #     return sorted(chan_list, key=lambda it: self._fluor_order.index(it))
    #
    # @property
    # def n_channels(self) -> int:
    #     """number of fluorescence channels"""
    #     return len(self.channels)
    #
    # @property
    # def sources(self) -> list[Source]:
    #     """list of unique sources"""
    #     return list(self.parsed_df['source'].unique().sort())
    #
    # @property
    # def n_sources(self) -> int:
    #     """number of source name"""
    #     return len(self.sources)
    #
    # @property
    # def areas(self) -> list[Area]:
    #     """list of area in the given ``classified_column``"""
    #     return list(self.parsed_df[self.classified_column].unique())
    #
    # @property
    # def n_total_rois(self) -> int:
    #     """number of total rois"""
    #     return self.parsed_df.shape[0]
    #
    # @property
    # def nroi_channel(self) -> pl.DataFrame:
    #     """channels and counts Dataframe"""
    #     return self.parsed_df['channel'].value_counts()
    #
    # @property
    # def nroi_source(self) -> pl.DataFrame:
    #     """channels and counts Dataframe"""
    #     return self.parsed_df['source'].value_counts()
    #
    # @property
    # def nroi_area(self) -> pl.DataFrame:
    #     """dict with key (area name) and value (number of cells)"""
    #     return self.parsed_df[self.classified_column].value_counts()

    # ========================= #
    # Classifier Output Methods #
    # ========================= #

    # def get_percent_sorted_df(self, hemisphere: HEMISPHERE_TYPE = 'both') -> pl.DataFrame:
    #     """
    #     Get percentage-sorted dataframe
    #
    #     :param hemisphere: which hemisphere {'ipsi', 'contra', 'both'}
    #     :return: dataframe
    #
    #     **hemi**: optional col. if hemisphere is`both`, then sum together
    #
    #     Example of merge level equal to 2::
    #
    #         ┌─────────┬────────────┬────────┬────────┬───────────┐
    #         │ source  ┆ merge_ac_2 ┆ *hemi. ┆ n_rois ┆ percent   │
    #         │ ---     ┆ ---        ┆ ---    ┆ ---    ┆ ---       │
    #         │ str     ┆ str        ┆ str    ┆ u32    ┆ f64       │
    #         ╞═════════╪════════════╪════════╪════════╪═══════════╡
    #         │ pRSC    ┆ VIS        ┆ ipsi   ┆ 5701   ┆ 22.761209 │
    #         │ aRSC    ┆ MO         ┆ ipsi   ┆ 2277   ┆ 21.360225 │
    #         │ overlap ┆ MO         ┆ ipsi   ┆ 571    ┆ 17.799252 │
    #         │ aRSC    ┆ ACA        ┆ ipsi   ┆ 1729   ┆ 16.219512 │
    #         │ overlap ┆ SUB        ┆ ipsi   ┆ 494    ┆ 15.399002 │
    #         │ …       ┆ …          ┆ …      ┆ …      ┆ …         │
    #         │ pRSC    ┆ MED        ┆ contra ┆ 1      ┆ 0.003992  │
    #         │ pRSC    ┆ PPN        ┆ contra ┆ 1      ┆ 0.003992  │
    #         │ pRSC    ┆ APr        ┆ ipsi   ┆ 1      ┆ 0.003992  │
    #         │ pRSC    ┆ BMA        ┆ contra ┆ 1      ┆ 0.003992  │
    #         │ pRSC    ┆ P5         ┆ contra ┆ 1      ┆ 0.003992  │
    #         └─────────┴────────────┴────────┴────────┴───────────┘
    #     """
    #
    #     if hemisphere in ('contra', 'ipsi'):
    #         cols = ['source', self.classified_column, 'hemi.']
    #     else:
    #         cols = ['source', self.classified_column]
    #
    #     df = self.parsed_df
    #     df = (df.select(cols)
    #           .group_by(cols)
    #           .agg(pl.col(self.classified_column).count().alias('n_rois'))
    #           .with_columns((pl.col('n_rois') / pl.col('n_rois').sum().over('source') * 100).alias('percent'))
    #           .sort('percent', descending=True))
    #
    #     return df

    # def get_classified_data(
    #         self,
    #         norm: MouseBrainRoiNormHandler | None = None, *,
    #         top_area: int | None = None,
    #         source: Source | None = None,
    #         add_other: bool = False,
    #         supply_overlap: bool = True,
    #         hemisphere: HEMISPHERE_TYPE = 'both',
    #         area: Area | list[Area] | None = None
    # ) -> 'RoiClassifiedNormTable':
    #     """
    #     processed data for plotting / visualization
    #
    #     :param norm: :class:`BrainMapNormHandler`
    #     :param top_area: select top ranks area based on channel-based normalized percentage
    #     :param source: specify source, if not then produce all source
    #     :param add_other: the rest of regions (after top selection), classified as `other` (i.e., pie chart)
    #     :param supply_overlap: add overlap roi counts into other channel(s)
    #     :param hemisphere: filter the output data with selected hemisphere {'ipsi', 'contra', 'both'}
    #     :param area: filter the output data with selected area
    #     :return: :class:`RoiClassifiedNormTable`
    #     """
    #     if supply_overlap and source != 'overlap':
    #         sup_source = list(self.fluor_repr.values())
    #         sup_source.remove('overlap')
    #         supply_df = supply_overlap_dataframe(self.parsed_df, supply_channel_name=sup_source)
    #         self.parsed_df = supply_df  # trigger setter
    #     else:
    #         Logger.warning('Source counts exclude overlap channel!')
    #
    #     df = self.get_percent_sorted_df(hemisphere)
    #
    #     #
    #     if source is not None:
    #         df = df.filter(pl.col('source') == source)
    #
    #     #
    #     if top_area is not None:
    #         top_area = int(top_area)
    #         if source is not None:
    #             df = self._select_top_region_single_channel(df, top_area, source, add_other)
    #         else:
    #             df = self._select_top_region_all_channel(df, top_area)
    #
    #     #
    #     if norm is not None:
    #         if norm.norm_type is not None:
    #             df = norm.expand(df, area_col=self.classified_column, expand_cols=['n_rois'])
    #
    #         if norm.norm_type in ('cell', 'volume'):
    #             df = norm.handle_failure(df)
    #
    #     norm_type = norm.norm_type if norm is not None else None
    #
    #     #
    #     df = df.with_columns(pl.lit(self.ccf_dir.animal).alias('animal'))
    #
    #     #
    #     ret = RoiClassifiedNormTable(self, norm_type, hemisphere, df)
    #
    #     if hemisphere != 'both':
    #         Logger.info(f'Filter classified result from: {hemisphere} hemisphere')
    #         ret = ret.with_hemisphere(hemisphere)
    #
    #     if area is not None:
    #         ret = ret.with_areas(area)
    #
    #     return ret


# ================ #
# Classified Table #
# ================ #

@attrs.define
class RoiClassifiedNormTable:
    source_classifier: RoiClassifier

    norm_type: ROIS_NORM_TYPE | None = attrs.field(
        validator=attrs.validators.optional(attrs.validators.in_(get_args(ROIS_NORM_TYPE)))
    )

    hemisphere: HEMISPHERE_TYPE = attrs.field(validator=attrs.validators.in_(get_args(HEMISPHERE_TYPE)))

    data: pl.DataFrame
    """processed data after ROIClassifier

    * Optional col: hemi. , only existed if init with either ``hemisphere`` is 'ipsi' or 'contra' 

    * <ROIS_NORM_TYPE>_norm depending on ``norm_type``

    Example of data with volume normalization method::

        ┌─────────┬────────────┬────────┬───────────┬───┬───────────┬───────────┬─────────────────┬────────┐
        │ channel ┆ merge_ac_2 ┆ n_rois ┆ percent   ┆ … ┆ Volumes   ┆ n_neurons ┆ *volume_norm_n_r┆ animal │
        │ ---     ┆ ---        ┆ ---    ┆ ---       ┆   ┆ [mm^3]    ┆ ---       ┆ ois             ┆ ---    │
        │ str     ┆ str        ┆ i64    ┆ f64       ┆   ┆ ---       ┆ i64       ┆ ---             ┆ str    │
        │         ┆            ┆        ┆           ┆   ┆ f64       ┆           ┆ f64             ┆        │
        ╞═════════╪════════════╪════════╪═══════════╪═══╪═══════════╪═══════════╪═════════════════╪════════╡
        │ overlap ┆ ACA        ┆ 423    ┆ 30.344333 ┆ … ┆ 5.222484  ┆ 337372    ┆ 80.995934       ┆ YW051  │
        │ gfp     ┆ MO         ┆ 3352   ┆ 24.545987 ┆ … ┆ 22.248234 ┆ 985411    ┆ 150.663641      ┆ YW051  │
        │ rfp     ┆ ACA        ┆ 1383   ┆ 23.791502 ┆ … ┆ 5.222484  ┆ 337372    ┆ 264.816494      ┆ YW051  │
        │ gfp     ┆ ACA        ┆ 3130   ┆ 22.920328 ┆ … ┆ 5.222484  ┆ 337372    ┆ 599.331616      ┆ YW051  │
        │ …       ┆ …          ┆ …      ┆ …         ┆ … ┆ …         ┆ …         ┆ …               ┆ …      │
        │ overlap ┆ SS         ┆ 1      ┆ 0.071736  ┆ … ┆ 37.177937 ┆ 2384622   ┆ 0.026898        ┆ YW051  │
        │ overlap ┆ ECT        ┆ 1      ┆ 0.071736  ┆ … ┆ 3.457703  ┆ 387378    ┆ 0.289209        ┆ YW051  │
        │ overlap ┆ TEa        ┆ 1      ┆ 0.071736  ┆ … ┆ 3.860953  ┆ 386396    ┆ 0.259003        ┆ YW051  │
        │ rfp     ┆ TT         ┆ 1      ┆ 0.017203  ┆ … ┆ 1.734078  ┆ 124596    ┆ 0.576675        ┆ YW051  │
        └─────────┴────────────┴────────┴───────────┴───┴───────────┴───────────┴─────────────────┴────────┘
    """

    @property
    def value_col(self) -> str:
        """column name for the value"""
        if self.norm_type in ('cell', 'volume'):
            return f'{self.norm_type}_norm_n_rois'
        elif self.norm_type == 'channel':
            return 'percent'
        else:
            return 'n_rois'


# ======================================== #
# Concat CSV (Add Source and Channel Info) #
# ======================================== #


# noinspection PyTypeChecker
def _concat_channel(ccf_dir: AbstractCCFDir,
                    fluor_repr: FluorReprType,
                    plane: PLANE_TYPE) -> pl.DataFrame:
    """
    Find the csv data from `labelled_roi_folder`, if multiple files are found, concat to single df.
    `channel` & `source` columns are added to the dataframe.

    If sagittal slice, auto move ipsi/contra hemispheres dataset (`resize_ipsi`, `resize_contra`)
    to new `resize` directory

    :param ccf_dir: :class:`~neuralib.atlas.ccf.core.AbstractCCFDir()`
    :param fluor_repr: ``FluorReprType``
    :param plane: ``PLANE_TYPE`` {'coronal', 'sagittal', 'transverse'}
    :return:
    """
    if plane == 'sagittal':
        _auto_sagittal_combine(ccf_dir)
    elif plane == 'coronal':
        _auto_coronal_combine(ccf_dir)


def _auto_overlap_copy(ccf: CoronalCCFOverlapDir | SagittalCCFOverlapDir) -> None:
    src = uglob(ccf.labelled_roi_folder_overlap, '*.csv')
    filename = f'{ccf.animal}_overlap_roitable'
    if ccf.plane_type == 'sagittal':
        filename += f'_{ccf.hemisphere}'

    dst = (ccf.labelled_roi_folder / filename).with_suffix('.csv')
    shutil.copy(src, dst)
    Logger.log(LOGGING_IO_LEVEL, f'copy overlap file from {src} to {dst}')


def _auto_coronal_combine(ccf_dir: CoronalCCFDir | CoronalCCFOverlapDir):
    _auto_overlap_copy(ccf_dir)


def _auto_sagittal_combine(ccf_dir: SagittalCCFDir | SagittalCCFOverlapDir) -> None:
    """copy file from overlap dir to major fluorescence (channel) folder,
    then combine different hemisphere data"""

    old_args = ccf_dir.hemisphere

    def with_hemisphere_stem(ccf: SagittalCCFDir | SagittalCCFOverlapDir) -> list[Path]:
        ls = list(ccf.labelled_roi_folder.glob('*.csv'))
        for it in ls:
            if ccf.hemisphere not in it.name:
                new_path = it.with_stem(it.stem + f'_{ccf.hemisphere}')
                it.rename(new_path)

        return list(ccf.labelled_roi_folder.glob('*.csv'))  # new glob

    mv_list = []

    ccf_dir.hemisphere = 'ipsi'
    if isinstance(ccf_dir, SagittalCCFOverlapDir):
        _auto_overlap_copy(ccf_dir)
    ext = with_hemisphere_stem(ccf_dir)
    mv_list.extend(ext)

    #
    ccf_dir.hemisphere = 'contra'
    if isinstance(ccf_dir, SagittalCCFOverlapDir):
        _auto_overlap_copy(ccf_dir)
    ext = with_hemisphere_stem(ccf_dir)
    mv_list.extend(ext)

    #
    ccf_dir.hemisphere = 'both'  # as resize
    target = ccf_dir.labelled_roi_folder
    for file in mv_list:
        shutil.copy(file, target / file.name)
        Logger.log(LOGGING_IO_LEVEL, f'copy file from {file} to {target / file.name}')

    ccf_dir.hemisphere = old_args  # assign back
