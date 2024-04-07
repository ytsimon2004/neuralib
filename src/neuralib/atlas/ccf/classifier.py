from __future__ import annotations

from pathlib import Path
from typing import TypedDict, Final, Literal, get_args

import attrs
import numpy as np
import polars as pl
from typing_extensions import Self, TypeAlias

from neuralib.atlas.ccf.core import CCFBaseDir
from neuralib.atlas.ccf.norm import MouseBrainRoiNormHandler, ROIS_NORM_TYPE
from neuralib.atlas.map import merge_until_level, NUM_MERGE_LAYER, DEFAULT_FAMILY_DICT
from neuralib.atlas.type import Area, HEMISPHERE_TYPE, Source, Channel
from neuralib.atlas.util import PLANE_TYPE
from neuralib.util.color_logging import setup_clogger, LOGGING_IO_LEVEL

__all__ = [
    'FluorReprType',
    #
    'UserInjectionConfig',
    'CLASSIFIED_COL',
    'RoiClassifier',
    'RoiClassifiedNormTable',
    #
    'supply_overlap_dataframe',
    #
    '_concat_channel',
    'parse_csv'
]

Logger = setup_clogger(caller_name=Path(__file__).name)

FluorReprType: TypeAlias = dict[Channel, Source]


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


# Example (replace to user-specific)
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

    def __init__(self, ccf_dir: CCFBaseDir,
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

    def _cache_parsed_dataframe(self, force_save: bool = False) -> pl.DataFrame:
        """concat & parse & add fields and cache a parsed dataframe"""
        file = self.ccf_dir.parse_csv

        if not file.exists() or force_save:
            df = parse_csv(self.ccf_dir, self.fluor_repr, plane=self.plane)

            ac = df['acronym']

            # add merge level cols
            df = df.with_columns(
                pl.Series(name=f'merge_ac_{level}', values=merge_until_level(ac, level))
                for level in range(NUM_MERGE_LAYER)
            )

            # add family col
            def categorize_family(row) -> str:
                for name, family in DEFAULT_FAMILY_DICT.items():
                    if row in family:
                        return name

            df = df.with_columns(pl.col('merge_ac_0').apply(categorize_family)
                                 .fill_null('unknown')
                                 .alias('family'))

            df.write_csv(file)
        else:
            df = pl.read_csv(file)

        return df

    def _ignore_injection_site(self, df: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
        area = self._injection_area
        hemi = self._injection_hemi
        Logger.info(f'remove {area} in {hemi} in {self.ccf_dir.animal}')

        expr1 = pl.col('acronym').str.starts_with(area)
        expr2 = pl.col('hemi.') == hemi

        if verbose:
            exc = (
                df.filter(pl.col('acronym').str.starts_with(area))
                .groupby('hemi.').agg(pl.count())
            )

            print(exc)

        return df.filter(~(expr1 & expr2))

    # ===== #
    # Utils #
    # ===== #

    # noinspection PyTypeChecker
    @property
    def classified_column(self) -> CLASSIFIED_COL:
        """ """
        if self.merge_level is None:
            return 'acronym_abbr'

        if isinstance(self.merge_level, int):
            return f'merge_ac_{self.merge_level}'
        elif isinstance(self.merge_level, str) and self.merge_level in self.parsed_df.columns:
            return self.merge_level
        else:
            raise ValueError(f'invalid merge level: {self.merge_level}')

    @property
    def channels(self) -> list[Channel]:
        """list of fluorescence channels
        (sorted based on :class:`~neuralib.atlas.ccf.classifier.UserInjectionConfig.fluor_repr`)"""
        chan_list = self.parsed_df['channel'].unique()
        return sorted(chan_list, key=lambda it: self._fluor_order.index(it))

    @property
    def n_channels(self) -> int:
        """number of fluorescence channels"""
        return len(self.channels)

    @property
    def sources(self) -> list[Source]:
        """list of unique sources"""
        return list(self.parsed_df['source'].unique().sort())

    @property
    def n_sources(self) -> int:
        """number of source name"""
        return len(self.sources)

    @property
    def areas(self) -> list[Area]:
        """list of area in the given ``classified_column``"""
        return list(self.parsed_df[self.classified_column].unique())

    @property
    def n_total_rois(self) -> int:
        """number of total rois"""
        return self.parsed_df.shape[0]

    @property
    def nroi_channel(self) -> pl.DataFrame:
        """channels and counts Dataframe"""
        return self.parsed_df['channel'].value_counts()

    @property
    def nroi_source(self) -> pl.DataFrame:
        """channels and counts Dataframe"""
        return self.parsed_df['source'].value_counts()

    @property
    def nroi_area(self) -> pl.DataFrame:
        """dict with key (area name) and value (number of cells)"""
        return self.parsed_df[self.classified_column].value_counts()

    # ========================= #
    # Classifier Output Methods #
    # ========================= #

    def get_percent_sorted_df(self, hemisphere: HEMISPHERE_TYPE = 'both') -> pl.DataFrame:
        """
        Get percentage-sorted dataframe

        :param hemisphere: which hemisphere {'ipsi', 'contra', 'both'}
        :return: dataframe

        *hemi: optional col. if hemisphere is`both`, then sum together

        Example of merge level equal to 2::

            ┌─────────┬────────────┬────────┬────────┬───────────┐
            │ source  ┆ merge_ac_2 ┆ *hemi. ┆ n_rois ┆ percent   │
            │ ---     ┆ ---        ┆ ---    ┆ ---    ┆ ---       │
            │ str     ┆ str        ┆ str    ┆ u32    ┆ f64       │
            ╞═════════╪════════════╪════════╪════════╪═══════════╡
            │ pRSC    ┆ VIS        ┆ ipsi   ┆ 5701   ┆ 22.761209 │
            │ aRSC    ┆ MO         ┆ ipsi   ┆ 2277   ┆ 21.360225 │
            │ overlap ┆ MO         ┆ ipsi   ┆ 571    ┆ 17.799252 │
            │ aRSC    ┆ ACA        ┆ ipsi   ┆ 1729   ┆ 16.219512 │
            │ overlap ┆ SUB        ┆ ipsi   ┆ 494    ┆ 15.399002 │
            │ …       ┆ …          ┆ …      ┆ …      ┆ …         │
            │ pRSC    ┆ MED        ┆ contra ┆ 1      ┆ 0.003992  │
            │ pRSC    ┆ PPN        ┆ contra ┆ 1      ┆ 0.003992  │
            │ pRSC    ┆ APr        ┆ ipsi   ┆ 1      ┆ 0.003992  │
            │ pRSC    ┆ BMA        ┆ contra ┆ 1      ┆ 0.003992  │
            │ pRSC    ┆ P5         ┆ contra ┆ 1      ┆ 0.003992  │
            └─────────┴────────────┴────────┴────────┴───────────┘
        """

        if hemisphere in ('contra', 'ipsi'):
            cols = ['source', self.classified_column, 'hemi.']
        else:
            cols = ['source', self.classified_column]

        df = self.parsed_df
        df = (df.select(cols)
              .group_by(cols)
              .agg(pl.col(self.classified_column).count().alias('n_rois'))
              .with_columns((pl.col('n_rois') / pl.col('n_rois').sum().over('source') * 100).alias('percent'))
              .sort('percent', descending=True))

        return df

    def get_classified_data(
            self,
            norm: MouseBrainRoiNormHandler | None = None, *,
            top_area: int | None = None,
            source: Source | None = None,
            add_other: bool = False,
            supply_overlap: bool = True,
            hemisphere: HEMISPHERE_TYPE = 'both',
            area: Area | list[Area] | None = None
    ) -> RoiClassifiedNormTable:
        """
        processed data for plotting / visualization

        :param norm: :class:`BrainMapNormHandler`
        :param top_area: select top ranks area based on channel-based normalized percentage
        :param source: specify source, if not then produce all source
        :param add_other: the rest of regions (after top selection), classified as `other` (i.e., pie chart)
        :param supply_overlap: add overlap roi counts into other channel(s)
        :param hemisphere: filter the output data with selected hemisphere {'ipsi', 'contra', 'both'}
        :param area: filter the output data with selected area
        :return: :class:`RoiClassifiedNormTable`
        """
        if supply_overlap and source != 'overlap':
            supply_df = supply_overlap_dataframe(self.parsed_df)
            self.parsed_df = supply_df  # trigger setter
        else:
            Logger.warning(f'Source counts exclude overlap channel!')

        df = self.get_percent_sorted_df(hemisphere)

        #
        if source is not None:
            df = df.filter(pl.col('source') == source)

        #
        if top_area is not None:
            top_area = int(top_area)
            if source is not None:
                df = self._select_top_region_single_channel(df, top_area, source, add_other)
            else:
                df = self._select_top_region_all_channel(df, top_area)

        #
        if norm is not None:
            if norm.norm_type is not None:
                df = norm.expand(df, area_col=self.classified_column, expand_cols=['n_rois'])

            if norm.norm_type in ('cell', 'volume'):
                df = norm.handle_failure(df)

        norm_type = norm.norm_type if norm is not None else None

        #
        df = df.with_columns(pl.lit(self.ccf_dir.animal).alias('animal'))

        #
        ret = RoiClassifiedNormTable(self, norm_type, hemisphere, df)

        if hemisphere != 'both':
            Logger.info(f'Filter classified result from: {hemisphere} hemisphere')
            ret = ret.with_hemisphere(hemisphere)

        if area is not None:
            ret = ret.with_areas(area)

        return ret

    def _select_top_region_single_channel(self,
                                          df: pl.DataFrame,
                                          top_area: int,
                                          source: Source,
                                          add_other: bool = False) -> pl.DataFrame:
        if top_area > df.shape[0]:
            Logger.warning(f'only {df.shape[0]} areas are classified,'
                           f'{top_area} areas exceed, thus use all areas instead')
        else:
            df = df[:top_area]

        if add_other and (100 - df['percent'].sum()) > 0:
            other_perc = max(0, 100 - df['percent'].sum())
            total_num = self.nroi_source.filter(pl.col('source') == source)['count'].item()
            other_num = total_num - df['n_rois'].sum()

            schema = {
                df.columns[i]: df.dtypes[i]
                for i in range(df.shape[1])
            }
            row = pl.DataFrame([[source, 'other', other_num, other_perc]], schema=schema)  # `other` row

            df = pl.concat([df, row])

        return df

    def _select_top_region_all_channel(self, df: pl.DataFrame, top_area: int) -> pl.DataFrame:
        ref_df = df[:top_area]
        _region = ref_df[self.classified_column].unique()

        return df.filter(pl.col(self.classified_column).is_in(_region))


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

    @property
    def classified_column(self) -> CLASSIFIED_COL:
        """classified column based on the ``RoiClassifier``"""
        return self.source_classifier.classified_column

    def with_areas(self, areas: Area | list[Area]) -> Self:
        """
        filter the data with selected area

        :param areas: area or list of area
        :return: :class:`RoiClassifiedNormTable`
        """
        if isinstance(areas, str):
            areas = [areas]

        _data = self.data.filter(pl.col(self.classified_column).is_in(areas))

        if _data.is_empty():
            raise ValueError(f'{areas} not found')

        return attrs.evolve(self, data=_data)

    def with_hemisphere(self, hemi: HEMISPHERE_TYPE) -> Self:
        """
        filter the data with selected hemisphere

        :param hemi: {'ipsi', 'contra'}
        :return: :class:`RoiClassifiedNormTable`
        """
        if hemi not in ('ipsi', 'contra'):
            raise ValueError('')

        return attrs.evolve(self,
                            hemisphere=hemi,
                            data=self.data.filter(pl.col('hemi.') == hemi))

    def get_bias_value_dataframe(self,
                                 yunit: str = 'percent',
                                 to_index=True,
                                 verbose=True) -> pl.DataFrame:
        """
        Dataframe with `bias value` using either

        1. simple subtraction using either *percent* or *n_rois* in `self.data` (bias_value)

        2. index calculation (bias index)

        :param yunit:
        :param to_index: to bias index (1og2 (P_areaA / P_areaB)). refer to Chen et al., 2022. biorxiv.
            otherwise, do simple subtraction
            NOTE that index is currently calculate based on `channel-normalization` value (percent col)
        :param verbose: do print for output
        :return: DataFrame

        Example::

            ┌────────────┬────────────┐
            │ merge_ac_1 ┆ bias_value*│
            │ ---        ┆ ---        │
            │ str        ┆ f64        │
            ╞════════════╪════════════╡
            │ MO         ┆ -18.591053 │
            │ SS         ┆ -5.362179  │
            │ …          ┆ …          │
            │ CP         ┆ 3.897736   │
            │ VIS        ┆ 12.899966  │
            └────────────┴────────────┘
        """
        bias_col = 'bias_value'

        if to_index:
            expr_calc = (pl.col('pRSC') / pl.col('aRSC')).apply(np.log2)
            yunit = 'percent'  # force
            bias_col = 'bias_index'
        else:
            expr_calc = (pl.col('pRSC') - pl.col('aRSC'))

        df = (
            self.data
            .select(self.classified_column, 'source', yunit)
            .sort(self.classified_column, 'source')
            .pivot(values=yunit, index=self.classified_column, columns='source', aggregate_function='first')
            .fill_null(0)
            .with_columns(expr_calc.alias(bias_col))
            .filter(~pl.col(bias_col).is_infinite())  # log2 index calc fail
            .select(self.classified_column, bias_col)
            .sort(by=bias_col)
        )

        if verbose:
            from neuralib.util.util_verbose import printdf
            printdf(df)

        return df

    def to_bokeh(self, yunit: str = 'percent') -> dict[str, list]:
        """
        to bokeh data ColumnSource input structure

        :param yunit: {'percent', 'n_rois'}
        :return:
            key: `x` & `counts`

            value: (area, channel) & neuronal counts or percentage
        """
        if yunit == 'n_rois':
            icol = 2
        elif yunit == 'percent':
            icol = 3
        else:
            raise ValueError(f'unknown yunit: {yunit}')

        data = self.data.sort('source')

        ret = {'x': [], 'counts': []}
        for row in data.rows():
            ret['x'].append((row[1], row[0]))
            ret['counts'].append(row[icol])

        return ret

    def to_winner_dataframe(self) -> pl.DataFrame:
        """
        for ternary plot in plotly module

        :return: dataframe
        ::

            ┌────────────┬─────────┬──────┬──────┬───────┬────────┐
            │ merge_ac_1 ┆ overlap ┆ pRSC ┆ aRSC ┆ total ┆ winner │
            │ ---        ┆ ---     ┆ ---  ┆ ---  ┆ ---   ┆ ---    │
            │ str        ┆ i64     ┆ i64  ┆ i64  ┆ i64   ┆ str    │
            ╞════════════╪═════════╪══════╪══════╪═══════╪════════╡
            │ ACA        ┆ 1000    ┆ 1048 ┆ 2865 ┆ 4913  ┆ gfp    │
            │ RSP        ┆ 327     ┆ 1086 ┆ 1345 ┆ 2758  ┆ gfp    │
            │ …          ┆ …       ┆ …    ┆ …    ┆ …     ┆ …      │
            │ CB         ┆ 0       ┆ 0    ┆ 1    ┆ 1     ┆ gfp    │
            │ CUN        ┆ 0       ┆ 0    ┆ 1    ┆ 1     ┆ gfp    │
            └────────────┴─────────┴──────┴──────┴───────┴────────┘
        """
        df = (self.data
              .pivot(values='n_rois', columns='source', index=self.classified_column, aggregate_function='first')
              .fill_nan(0)
              .fill_null(0)
              .with_columns((pl.col('pRSC') + pl.col('aRSC') + pl.col('overlap')).alias('total')))

        cols = self.source_classifier.sources
        region_counts = df.select(cols).to_numpy()
        winner_idx = np.argmax(region_counts, axis=1).astype(int)
        df = df.with_columns(pl.Series([cols[idx] for idx in winner_idx]).alias('winner'))

        return df

    def write_csv(self, out: Path,
                  animal_id_col: str | None = None):
        data = self.data
        if animal_id_col is not None:
            data = data.with_columns(pl.lit(animal_id_col).alias('animal_id'))

        data.write_csv(out)


# ======================================== #
# Concat CSV (Add Source and Channel Info) #
# ======================================== #


def _concat_channel(ccf_dir: CCFBaseDir,
                    fluor_repr: FluorReprType,
                    plane: PLANE_TYPE) -> pl.DataFrame:
    """
    Find the csv data from `labelled_roi_folder`, if multiple files are found, concat to single df.
    `channel` & `source` columns are added to the dataframe

    :param ccf_dir: :class:`~neuralib.atlas.ccf.core.CCFBaseDir()`
    :param fluor_repr: ``FluorReprType``
    :param plane: ``PLANE_TYPE`` {'coronal', 'sagittal', 'transverse'}
    :return:
    """
    f = list(ccf_dir.labelled_roi_folder.glob('*.csv'))

    if len(f) == 1:
        return _single_proc(f, fluor_repr)

    else:
        return _multiple_concat_proc(f, plane, ccf_dir, fluor_repr)


def _single_proc(f: list[Path], fluor_repr: FluorReprType):
    Logger.log(LOGGING_IO_LEVEL, f'load single csv file')
    file = f[0]
    df = pl.read_csv(file)
    for pattern in ['rfp', 'gfp', 'overlap']:
        if pattern in file.name:
            channel = pattern
            source = fluor_repr[channel]

            return (df.with_columns(pl.lit(channel).alias('channel'))
                    .with_columns(pl.lit(source).alias('source')))

    raise RuntimeError('')


def _multiple_concat_proc(f: list[Path],
                          plane: PLANE_TYPE,
                          ccf_dir: CCFBaseDir,
                          fluor_repr: FluorReprType) -> pl.DataFrame:
    Logger.log(LOGGING_IO_LEVEL, f'load multiple csv files: {len(f)} files')

    df_list = []
    for ff in f:
        if 'rfp' in ff.name:
            channel = 'rfp'
        elif 'gfp' in ff.name:
            channel = 'gfp'
        elif 'overlap' in ff.name:
            channel = 'overlap'
        else:
            continue

        source = fluor_repr[channel]

        df_list.append(pl.read_csv(ff).with_columns(pl.lit(channel).alias('channel'))
                       .with_columns(pl.lit(source).alias('source')))

    # validate
    if plane == 'coronal' and len(df_list) != 3:
        raise RuntimeError(f'missing csv in {ccf_dir.labelled_roi_folder} for {plane} pipeline')

    if plane == 'sagittal' and len(df_list) != 6:
        raise RuntimeError(f'missing csv in {ccf_dir.labelled_roi_folder} for {plane} pipeline')

    return pl.concat(df_list)


# ========================= #
# Raw CSV Parsing Functions #
# ========================= #

def parse_csv(ccf_dir: CCFBaseDir | None,
              fluor_repr: FluorReprType, *,
              plane: PLANE_TYPE = 'coronal',
              df: pl.DataFrame | None = None,
              invert_hemi: bool = False) -> pl.DataFrame:
    """
    Narrow down the info in the ccf roi output

    columns `abbr`, `acronym_abbr` and `hemi` are added to the df

    :param ccf_dir: if None, directly give ``df`` arg
    :param fluor_repr: ``FluorReprType``
    :param plane: ``PLANE_TYPE`` {'coronal', 'sagittal', 'transverse'}
    :param df: if None, give ``ccf_dir`` arg
    :param invert_hemi: if True, then ML_location >= 0 is ipsilateral site. otherwise, is contralateral site
    :return: DataFrame
    """

    if df is None:
        df = _concat_channel(ccf_dir, fluor_repr, plane)

    # pick up acronym with capital
    df = df.filter(pl.col('acronym').str.contains(r'[A-Z]+'))

    df = df.with_columns(
        # exclude words after `area` in the name col
        pl.col('name').str.extract(r'(.+area).*').fill_null(pl.col('name')).alias('abbr'),
        # remove subregion and layer info
        pl.col('acronym').str.extract(r'(CA1|CA2|CA3|[A-Za-z]+).*').fill_null(pl.col('acronym')).alias('acronym_abbr'),
    )

    # add hemisphere info (ml > 0 represent image right site, thus ipsilateral)
    if invert_hemi:
        expr = pl.when(pl.col('ML_location') < 0)
    else:
        expr = pl.when(pl.col('ML_location') >= 0)

    df = df.with_columns(
        expr
        .then(pl.lit('ipsi'))
        .otherwise(pl.lit('contra'))
        .alias('hemi.')
    )

    return df


# TODO might be user-specific, be more generalized
def supply_overlap_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Supply overlap counting in the parsed dataframe.

    ** Only used if excluding the overlap cell while counting the rfp and gfp channel


    :param df: dataframe contain `source` column with `overlap` literal in the cell
    """
    ori = df
    dat_s1 = df.filter(pl.col('source') == 'overlap').with_columns(pl.lit('aRSC').alias('source'))
    dat_s2 = df.filter(pl.col('source') == 'overlap').with_columns(pl.lit('pRSC').alias('source'))
    return pl.concat([ori, dat_s1, dat_s2])
