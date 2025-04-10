from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import polars as pl
from polars.polars import ColumnNotFoundError
from typing_extensions import Self

from neuralib.atlas.cellatlas import load_cellatlas
from neuralib.atlas.data import load_bg_volumes
from neuralib.atlas.map import merge_until_level, NUM_MERGE_LAYER, DEFAULT_FAMILY_DICT
from neuralib.atlas.typing import Channel, HEMISPHERE_TYPE
from neuralib.atlas.util import get_margin_merge_level
from neuralib.typing import PathLike
from neuralib.util.dataframe import DataFrameWrapper
from neuralib.util.utils import ensure_dir
from neuralib.util.verbose import print_save, print_load

__all__ = [
    'ROIS_NORM_TYPE',
    'RoiClassifierDataFrame',
    'RoiNormalizedDataFrame',
    'RoiSubregionDataFrame',
]

ROIS_NORM_TYPE = Literal['channel', 'volume', 'cell', 'none']
"""Roi normalized type"""


class RoiClassifierDataFrame(DataFrameWrapper):
    """
    RoiClassifierDataFrame with each roi (rows)

    - Required fields:
        - ``acronym`` - area acronym
        - ``AP_location``: anterior-posterior coordinates (mm)
        - ``DV_location``: dorsal-ventral coordinates (mm)
        - ``ML_location``: medial-lateral coordinates (mm)
        - ``channel``: fluorescence channel (i.e., gfp, rfp, mcherry, ...)
        - ``source``: source name (i.e., if circuit tracing, you can give source tracing area...)

    """
    _required_fields = ('acronym', 'AP_location', 'DV_location', 'ML_location', 'channel', 'source')
    _valid_classified_fields = ('acronym', 'tree_0', 'tree_1', 'tree_2', 'tree_3', 'tree_4', 'family')

    def __init__(self, df: pl.DataFrame, *,
                 cached_dir: PathLike | None = None,
                 invalid_post_processing_cache: bool = False):
        """

        :param df: DataFrame with required fields
        :param cached_dir: create cached directory
        :param invalid_post_processing_cache: invalid post processing cache if cached_dir is not None
        """
        self._df = df
        self._cached_dir = cached_dir
        self._invalid_post_processing_cache = invalid_post_processing_cache

        for field in self._required_fields:
            if field not in df.columns:
                raise RuntimeError(f'field not found: {field} -> {df.columns}')

        self.__allow_inplace = True

    def __repr__(self):
        return repr(self.dataframe())

    def dataframe(self, dataframe: pl.DataFrame = None, may_inplace=True):
        if dataframe is None:
            return self._df
        else:
            return RoiClassifierDataFrame(dataframe, cached_dir=self._cached_dir)

    @property
    def channels(self) -> list[Channel]:
        """list of channel names"""
        return self.dataframe()['channel'].unique().to_list()

    @property
    def n_channels(self) -> int:
        """number of channel"""
        return len(self.channels)

    @property
    def channel_counts(self) -> pl.DataFrame:
        """channel counts dataframe"""
        return self['channel'].value_counts()

    @property
    def source_counts(self) -> pl.DataFrame:
        """source counts dataframe"""
        return self['source'].value_counts()

    @property
    def sources(self) -> list[Channel]:
        """list of source names"""
        return self.dataframe()['source'].unique().to_list()

    @property
    def n_sources(self) -> int:
        """number of source"""
        return len(self.sources)

    @property
    def is_overlapped_channel(self) -> bool:
        """whether there is overlapped channel"""
        return 'overlap' in self.channels

    def get_channel_source_dict(self) -> dict[str, str]:
        """get channel (key): source (value) dict"""
        return {
            it[0]: it[1]
            for it in self['channel', 'source'].unique().iter_rows()
        }

    def get_classified_column(self, field: int | str | None, strict: bool = True) -> str:
        """classified column name, with the given

        :param field: int: ``tree_[level]``; str: ``get column``; None: ``acronym``
        :param strict: strict check for the pre-defined classified column
        """
        match field:
            case int():
                col = f'tree_{field}'
            case str():
                col = field
            case None:
                col = 'acronym'
            case _:
                raise TypeError(f'{type(field)}')

        if strict:
            if col not in self._valid_classified_fields:
                raise ValueError(f'invalid field: {field}')

        if col not in self.dataframe().columns:
            raise ColumnNotFoundError(f'{col} not found: {self.dataframe().columns}')

        return col

    def post_processing(self, *,
                        filter_capital: bool = True,
                        tree: bool = True,
                        family: bool = True,
                        hemisphere: bool = True,
                        copy_overlap: bool = True,
                        filter_injection: tuple[str, str] | None = None) -> Self:
        """
        Load the post-processing dataframe

        :param filter_capital: filter only ``acronym`` contain capital letters
        :param tree: with customized hierarchical tree structure based on allen brain
        :param family: with column ``family`` ('HB', 'HY', 'TH', 'MB', 'CB', 'CTXpl', 'HPF', 'ISOCORTEX', 'OLF', 'CTXsp'
        :param hemisphere: with column ``hemisphere`` with which hemisphere
        :param copy_overlap: Copy overlap channels counts to individual channels, only set true if overlapped roi is not counted for individual channels
        :param filter_injection:  filter out the injection site labelled
        :return:
        """
        if self._cached_dir is not None:
            file = ensure_dir(self._cached_dir) / 'parsed_roi.csv'

            if file.exists() and self._invalid_post_processing_cache:
                file.unlink()

            # load
            if file.exists():
                df = pl.read_csv(file)
                print_load(file)
                return RoiClassifierDataFrame(
                    df, cached_dir=self._cached_dir,
                    invalid_post_processing_cache=self._invalid_post_processing_cache
                )

            # write
            ret = self._post_processing(filter_capital, tree, family, hemisphere, copy_overlap, filter_injection)
            ret.dataframe().write_csv(file)
            print_save(file)
            return ret

        # return
        return self._post_processing(filter_capital, tree, family, hemisphere, copy_overlap, filter_injection)

    def _post_processing(self, filter_capital, tree, family, hemisphere, copy_overlap, filter_injection) -> Self:
        ret = self
        if filter_capital:
            ret = ret.filter_capital_name()
        if tree:
            ret = ret.with_tree_columns()
        if family:
            ret = ret.with_family_columns()
        if hemisphere:
            ret = ret.with_hemisphere_column()
        if copy_overlap:
            ret = ret.with_overlap_copy()
            self.__allow_inplace = False
        if filter_injection is not None:
            ret = ret.filter_injection_site(area=filter_injection[0], hemisphere=filter_injection[1])

        return ret

    def filter_injection_site(self, area: str, hemisphere: str) -> Self:
        """
        filter out the injection site labelled

        :param area: brain area
        :param hemisphere: which hemisphere
        :return:
        """
        expr1 = pl.col('acronym').str.starts_with(area)
        expr2 = pl.col('hemisphere') == hemisphere
        return self.filter(~(expr1 & expr2))

    def filter_capital_name(self) -> Self:
        """filter only ``acronym`` contain capital letters"""
        return self.filter(pl.col('acronym').str.contains(r'[A-Z]+'))

    def with_tree_columns(self) -> Self:
        """with customized hierarchical tree structure based on allen brain

        .. seealso::

            Wang et al 2020, https://doi.org/10.1016/j.cell.2020.04.007
        """
        acronym = self['acronym']
        return self.with_columns(
            pl.Series(name=f'tree_{level}', values=merge_until_level(acronym, level))
            for level in range(NUM_MERGE_LAYER)
        )

    def with_family_columns(self) -> Self:
        """with column ``family`` ('HB', 'HY', 'TH', 'MB', 'CB', 'CTXpl', 'HPF', 'ISOCORTEX', 'OLF', 'CTXsp')"""

        def get_family(row) -> str:
            for name, family in DEFAULT_FAMILY_DICT.items():
                if row in family:
                    return name
            return 'unknown'

        return self.with_columns(pl.col('tree_0').map_elements(get_family, return_dtype=pl.Utf8).alias('family'))

    def with_hemisphere_column(self, invert: bool = False) -> Self:
        """
        with column ``hemisphere`` with which hemisphere

        :param invert: invert hemisphere. Default ML >= 0 (ipsi), ML < 0 (contra)
        :return:
        """
        if invert:
            expr = pl.when(pl.col('ML_location') < 0)
        else:
            expr = pl.when(pl.col('ML_location') >= 0)

        return self.with_columns(expr.then(pl.lit('ipsi')).otherwise(pl.lit('contra')).alias('hemisphere'))

    def with_overlap_copy(self) -> Self:
        """Copy overlap channels counts to individual channels, only used if overlapped roi is not counted for individual channels"""
        if not self.__allow_inplace:
            raise RuntimeError('recurrent copy overlap')

        ret = [self.dataframe()]
        for channel, source in self.get_channel_source_dict().items():
            if channel not in 'overlap':
                df = (
                    self.dataframe().filter(pl.col('channel') == 'overlap')
                    .with_columns(pl.lit(channel).alias('channel'))
                    .with_columns(pl.lit(source).alias('source'))
                )
                ret.append(df)

        return RoiClassifierDataFrame(pl.concat(ret), cached_dir=self._cached_dir)

    # ==================== #
    # Normalized DataFrame #
    # ==================== #

    def to_normalized(self, norm: ROIS_NORM_TYPE,
                      level: int | str | None, *,
                      source: str | None = None,
                      top_area: int | None = None,
                      rest_as_others: bool = False,
                      hemisphere: HEMISPHERE_TYPE = 'both',
                      animal: str | None = None,
                      volume_norm_backend: Literal['cellatlas', 'brainglobe'] = 'cellatlas') -> RoiNormalizedDataFrame:
        """
        To the normalized dataframe (example as volume normalized) ::

            ┌─────────┬────────┬────────┬───────────┬────────────┬────────────────┬────────────┐
            │ source  ┆ tree_2 ┆ counts ┆ fraction  ┆ hemisphere ┆ Volumes [mm^3] ┆ normalized │
            │ ---     ┆ ---    ┆ ---    ┆ ---       ┆ ---        ┆ ---            ┆ ---        │
            │ str     ┆ str    ┆ u32    ┆ f64       ┆ str        ┆ f64            ┆ f64        │
            ╞═════════╪════════╪════════╪═══════════╪════════════╪════════════════╪════════════╡
            │ overlap ┆ ACA    ┆ 1208   ┆ 29.997517 ┆ both       ┆ 5.222484       ┆ 231.307537 │
            │ pRSC    ┆ ACA    ┆ 3296   ┆ 22.822324 ┆ both       ┆ 5.222484       ┆ 631.117254 │
            │ …       ┆ …      ┆ …      ┆ …         ┆ …          ┆ …              ┆ …          │
            │ pRSC    ┆ VIS    ┆ 4035   ┆ 27.939344 ┆ both       ┆ 12.957203      ┆ 311.409797 │
            │ overlap ┆ VIS    ┆ 628    ┆ 15.594736 ┆ both       ┆ 12.957203      ┆ 48.46725   │
            │ aRSC    ┆ VIS    ┆ 3865   ┆ 12.627005 ┆ both       ┆ 12.957203      ┆ 298.289682 │
            └─────────┴────────┴────────┴───────────┴────────────┴────────────────┴────────────┘


        :param norm: :attr:`~neuralib.atlas.ccf.classifier.ROIS_NORM_TYPE`
        :param level: tree level for determine which level of classified column
        :param source: filter only the given ``source``
        :param top_area: filter only the given top areas (sorted based on ``fraction``)
        :param rest_as_others: vertical concat a region called **other** with the roi filtered by ``top area``
        :param hemisphere:  filter only the given ``hemisphere``
        :param animal: with animal id column ``animal``
        :param volume_norm_backend: volume normalization backend. {'cellatlas', 'brainglobe'}
        :return: ``RoiNormalizedDataFrame``
        """
        cls_col = self.get_classified_column(level)

        if hemisphere != 'both':
            cols = ['source', cls_col, 'hemisphere']
        else:
            cols = ['source', cls_col]

        df = (
            self.dataframe().select(cols).group_by(cols).agg(pl.col(cls_col).count().alias('counts'))
            .with_columns((pl.col('counts') / pl.col('counts').sum().over('source') * 100).alias('fraction'))
            .sort('fraction', descending=True)
        )

        #
        if hemisphere != 'both':
            df = df.filter(pl.col('hemisphere') == hemisphere)
        else:
            df = df.with_columns(pl.lit(hemisphere).alias('hemisphere'))  # add back

        #
        if source is not None:
            df = df.filter(pl.col('source') == source)

        #
        if top_area is not None:
            if source is not None:
                df = self._filter_top_region_single_source(df, top_area, source, rest_as_others)
            else:
                df = self._filter_top_region_all_source(df, top_area, level)

        ret = RoiNormalizedDataFrame(df, cls_col, norm)

        #
        match norm:
            case 'volume':
                ret = ret.with_density_column(backend=volume_norm_backend)
            case 'cell':
                ret = ret.with_cell_density_column()
            case 'channel':
                ret = ret.with_columns(pl.col('fraction').alias('normalized'))  # copy from fraction
            case 'none':
                pass
            case _:
                raise ValueError(f'invalid norm method: {norm}')
        #
        if animal is not None:
            ret = ret.with_animal_column(animal)

        return ret

    def _filter_top_region_single_source(self, df, top_area, source, rest_as_others) -> pl.DataFrame:
        if top_area > df.shape[0]:
            print(f'{top_area} areas exceed, thus use all areas instead')
        else:
            df = df[:top_area]

        others = df['fraction'].sum()
        if rest_as_others and (100 - others) > 0:
            other_perc = max(0, 100 - others)
            total_counts = self.source_counts.filter(pl.col('source') == source)['count'].item()
            other_counts = total_counts - df['counts'].sum()

            schema = {df.columns[i]: df.dtypes[i] for i in range(df.shape[1])}
            row = pl.DataFrame([[source, 'other', other_counts, other_perc, df['hemisphere'][0]]], schema=schema,
                               orient='row')  # `other` row

            df = pl.concat([df, row])

        return df

    def _filter_top_region_all_source(self, df, top_area, level) -> pl.DataFrame:
        if top_area > df.shape[0]:
            print(f'{top_area} areas exceed, thus use all areas instead')
            ref_df = df
        else:
            ref_df = df[:top_area]

        cls_col = self.get_classified_column(level)
        region = ref_df[cls_col].unique()

        return df.filter(pl.col(cls_col).is_in(region))

    def to_subregion(self, region: str, *,
                     unit: Literal['counts', 'fraction'] = 'fraction',
                     source_order: tuple[str, ...] | None = None,
                     show_col: str | None = None,
                     animal: str | None = None) -> RoiSubregionDataFrame:
        """
        To the subregion dataframe (example as Visual region: VIS) ::

            ┌─────────┬───────────┬───────────┬───────────┬───┬──────────┬──────────┬──────────┬──────────┐
            │ source  ┆ VISam     ┆ VISp      ┆ VISpm     ┆ … ┆ VISal    ┆ VISpor   ┆ VISli    ┆ VISpl    │
            │ ---     ┆ ---       ┆ ---       ┆ ---       ┆   ┆ ---      ┆ ---      ┆ ---      ┆ ---      │
            │ str     ┆ f64       ┆ f64       ┆ f64       ┆   ┆ f64      ┆ f64      ┆ f64      ┆ f64      │
            ╞═════════╪═══════════╪═══════════╪═══════════╪═══╪══════════╪══════════╪══════════╪══════════╡
            │ overlap ┆ 39.649682 ┆ 15.127389 ┆ 28.025478 ┆ … ┆ 3.025478 ┆ 2.707006 ┆ 1.592357 ┆ 0.159236 │
            │ aRSC    ┆ 32.160414 ┆ 28.952135 ┆ 23.05304  ┆ … ┆ 6.080207 ┆ 1.293661 ┆ 2.069858 ┆ 0.07762  │
            │ pRSC    ┆ 25.947955 ┆ 27.95539  ┆ 27.459727 ┆ … ┆ 3.122677 ┆ 2.973978 ┆ 1.982652 ┆ 1.016109 │
            └─────────┴───────────┴───────────┴───────────┴───┴──────────┴──────────┴──────────┴──────────┘

        :param region: region name
        :param unit: value unit. {'counts', 'fraction'}. default is 'fraction'
        :param source_order: source order in dataframe (rows)
        :param show_col: force set show col to which level. use case: if a low level area name is classified and show in high level (i.e., TH).
        :param animal: with animal column as subregion dataframe
        :return:
        """

        _df = self.dataframe()
        source_order = source_order or tuple(_df['source'].unique().to_list())

        # query based on lowest tree level
        query_col = get_margin_merge_level(_df, region, 'lowest')
        result = _df.filter(pl.col(query_col) == region)

        # show based on highest tree level
        show_col = show_col or get_margin_merge_level(_df, region, 'highest')
        s, lv = show_col.rsplit('_', 1)
        show_col = f'{s}_{int(lv) + 1}'

        df = (result.select(['source', show_col])
              .group_by(['source', show_col])
              .agg(pl.col(show_col).count().alias('counts'))
              .with_columns((pl.col('counts') / pl.col('counts').sum().over('source') * 100).alias('fraction'))
              .sort('fraction', descending=True))

        # sort
        idx = {val: idx for idx, val in enumerate(source_order)}
        sort_expr = pl.col('source').replace(idx)

        # profile
        roi_profile = (
            df.group_by('source').agg(pl.col('counts').sum().alias('counts'))
            .join(_df.group_by('source').len(name='total'), on='source')
            .with_columns((pl.col('counts') / pl.col('total')).alias('total_fraction'))
            .sort(sort_expr)
        )

        # main result
        ret = (
            df.pivot(show_col, index='source', values=unit, aggregate_function='first')
            .fill_null(0)
            .sort(sort_expr)
        )

        subregion = RoiSubregionDataFrame(region, ret, roi_profile)

        if animal is not None:
            subregion = subregion.with_animal_column(animal)

        return subregion


class RoiNormalizedDataFrame(DataFrameWrapper):
    """
    RoiNormalizedDataFrame with each area per row (unique ``source``, ``hemisphere``)

    - Required fields:
        - ``counts``: roi counts
        - ``fraction``: roi fraction for individual sources (aka. per channel(source) normalized)
        - ``hemisphere``: which hemisphere
        - area column field {'acronym', 'tree_0', 'tree_1', 'tree_2', 'tree_3', 'tree_4'}

    - Optional field:
        - normalization-specific fields (if not 'none'): ``normalized``, ``Volumes [mm^3]``, ``volume_mm3``, ``n_neurons``

    """
    _required_fields = ('counts', 'fraction', 'hemisphere')

    def __init__(self, df: pl.DataFrame,
                 classified_column: str,
                 normalized: ROIS_NORM_TYPE):
        """
        :param df: DataFrame with required fields
        :param classified_column: classified column for the brain area
        :param normalized: :attr:`~neuralib.atlas.ccf.classifier.ROIS_NORM_TYPE`
        """

        for field in self._required_fields:
            if field not in df.columns:
                raise RuntimeError(f'field not found: {field}')

        self._df = df
        self._classified_column = classified_column
        self._normalized = normalized

    def __repr__(self):
        return repr(self.dataframe())

    @property
    def classified_column(self) -> str:
        """region classified column name"""
        return self._classified_column

    @property
    def normalized(self) -> ROIS_NORM_TYPE:
        """normalization type"""
        return self._normalized

    @property
    def value_column(self) -> str:
        """value column based on the ``normalized``"""
        match self._normalized:
            case 'volume' | 'cell' | 'channel':
                return 'normalized'
            case 'none':
                return 'counts'
            case _:
                raise ValueError(f'invalid normalized method: {self._normalized}')

    @property
    def normalized_unit(self) -> str:
        """unit based on the ``normalized``"""
        match self._normalized:
            case 'volume':
                return 'density (cells-mm3)'
            case 'cell':
                return 'cell density (%)'
            case 'channel':
                return 'fraction (%)'
            case 'none':
                return 'counts'
            case _:
                raise ValueError(f'invalid normalized unit: {self._normalized}')

    def dataframe(self, dataframe: pl.DataFrame = None, may_inplace=True):
        """
        RoiNormalizedDataFrame (Volume normalized as example) ::

            ┌─────────┬────────┬────────┬───────────┬────────────┬────────────────┬────────────┐
            │ source  ┆ tree_2 ┆ counts ┆ fraction  ┆ hemisphere ┆ Volumes [mm^3] ┆ normalized │
            │ ---     ┆ ---    ┆ ---    ┆ ---       ┆ ---        ┆ ---            ┆ ---        │
            │ str     ┆ str    ┆ u32    ┆ f64       ┆ str        ┆ f64            ┆ f64        │
            ╞═════════╪════════╪════════╪═══════════╪════════════╪════════════════╪════════════╡
            │ overlap ┆ ACA    ┆ 1208   ┆ 29.997517 ┆ both       ┆ 5.222484       ┆ 231.307537 │
            │ pRSC    ┆ ACA    ┆ 3296   ┆ 22.822324 ┆ both       ┆ 5.222484       ┆ 631.117254 │
            │ …       ┆ …      ┆ …      ┆ …         ┆ …          ┆ …              ┆ …          │
            │ pRSC    ┆ VIS    ┆ 4035   ┆ 27.939344 ┆ both       ┆ 12.957203      ┆ 311.409797 │
            │ overlap ┆ VIS    ┆ 628    ┆ 15.594736 ┆ both       ┆ 12.957203      ┆ 48.46725   │
            │ aRSC    ┆ VIS    ┆ 3865   ┆ 12.627005 ┆ both       ┆ 12.957203      ┆ 298.289682 │
            └─────────┴────────┴────────┴───────────┴────────────┴────────────────┴────────────┘

        """
        if dataframe is None:
            return self._df
        else:
            ret = RoiNormalizedDataFrame(dataframe, self._classified_column, self._normalized)
            return ret

    def with_density_column(self, backend: Literal['cellatlas', 'brainglobe'] = 'cellatas') -> Self:
        """

        :param backend: Volume information calculated from which backend. {'cellatlas', 'brainglobe'}
        :return:
        """
        match backend:
            case 'cellatlas':
                df_cellatlas = load_cellatlas().select('Volumes [mm^3]', 'acronym').rename({'acronym': self.classified_column})
                return (self.join(df_cellatlas, on=self.classified_column)
                        .with_columns((pl.col('counts') / pl.col('Volumes [mm^3]')).alias('normalized')))

            case 'brainglobe':
                df_bg_volume = load_bg_volumes().select('volume_mm3', 'acronym').rename({'acronym': self.classified_column})
                return (self.join(df_bg_volume, on=self.classified_column)
                        .with_columns((pl.col('counts') / pl.col('volume_mm3')).alias('normalized')))

            case _:
                raise ValueError(f'invalid backend: {backend}')

    def with_cell_density_column(self) -> Self:
        """Normalized to number of neurons foreach brain region (based on ``CellAtlas`` data source)"""
        df_cellatlas = load_cellatlas().select('n_neurons', 'acronym').rename({'acronym': self.classified_column})
        return (self.join(df_cellatlas, on=self.classified_column)
                .with_columns((pl.col('counts') / pl.col('n_neurons')).alias('normalized')))

    def with_animal_column(self, animal) -> Self:
        """with animal id column"""
        return self.with_columns(pl.lit(animal).alias('animal'))

    def filter_areas(self, areas: str | list[str]) -> Self:
        """filter the dataframe with specified areas"""
        if isinstance(areas, str):
            areas = [areas]

        ret = self.filter(pl.col(self.classified_column).is_in(areas))
        if ret.dataframe().is_empty():
            raise ValueError(f'{areas} not found')

        return ret

    def filter_sources(self, source: str | list[str]) -> Self:
        """filter the dataframe with specified sources"""
        if isinstance(source, str):
            source = [source]

        ret = self.filter(pl.col('source').is_in(source))
        if ret.dataframe().is_empty():
            raise ValueError(f'{source} not found')

        return ret

    def to_bias_index(self, source_a: str, source_b: str) -> pl.DataFrame:
        """
        Bias Index dataframe used to determine bias within two sources
        (positive value toward ``source a`` and negative value toward ``source b``) ::

            ┌────────┬────────────┐
            │ tree_2 ┆ bias_index │
            │ ---    ┆ ---        │
            │ str    ┆ f64        │
            ╞════════╪════════════╡
            │ ATN    ┆ -1.192889  │
            │ VIS    ┆ -1.145786  │
            │ CLA    ┆ -0.86059   │
            │ SUB    ┆ -0.478069  │
            │ STRd   ┆ -0.463589  │
            │ …      ┆ …          │
            │ ENT    ┆ 0.580593   │
            │ AUD    ┆ 0.610688   │
            │ PTLp   ┆ 1.292926   │
            │ MO     ┆ 1.945567   │
            │ SS     ┆ 2.163074   │
            └────────┴────────────┘

        :param source_a: source a string
        :param source_b: source b string
        :return:
        """
        expr_calc = (pl.col(source_a) / pl.col(source_b)).map_elements(np.log2, return_dtype=pl.Float64)
        df = (
            self.dataframe()
            .select(self.classified_column, 'source', 'fraction')
            .sort(self.classified_column, 'source')
            .pivot(values='fraction', index=self.classified_column, on='source', aggregate_function='first')
            .fill_null(0)
            .with_columns(expr_calc.alias('bias_index'))
            .filter(~pl.col('bias_index').is_infinite())  # log2 index calc fail
            .select(self.classified_column, 'bias_index')
            .sort(by='bias_index')
        )

        return df

    def to_winner(self, sources: Sequence[str]) -> pl.DataFrame:
        """
        Winner dataframe used for plotting (i.e., ternary plot) ::

            ┌────────┬─────────┬──────┬──────┬───────┬────────┐
            │ tree_2 ┆ overlap ┆ pRSC ┆ aRSC ┆ total ┆ winner │
            │ ---    ┆ ---     ┆ ---  ┆ ---  ┆ ---   ┆ ---    │
            │ str    ┆ u32     ┆ u32  ┆ u32  ┆ u32   ┆ str    │
            ╞════════╪═════════╪══════╪══════╪═══════╪════════╡
            │ ACA    ┆ 1208    ┆ 3296 ┆ 5761 ┆ 9057  ┆ aRSC   │
            │ VIS    ┆ 628     ┆ 4035 ┆ 3865 ┆ 7900  ┆ pRSC   │
            │ MO     ┆ 460     ┆ 714  ┆ 5829 ┆ 6543  ┆ aRSC   │
            │ …      ┆ …       ┆ …    ┆ …    ┆ …     ┆ …      │
            │ AUD    ┆ 44      ┆ 165  ┆ 534  ┆ 699   ┆ aRSC   │
            │ TEa    ┆ 34      ┆ 206  ┆ 358  ┆ 564   ┆ aRSC   │
            └────────┴─────────┴──────┴──────┴───────┴────────┘

        :param sources: source sequences for calculating the total. The above case should be specified as ['aRSC', 'pRSC']
        :return: Winner dataframe
        """
        df = (
            self.dataframe()
            .pivot(values=self.value_column, on='source', index=self.classified_column, aggregate_function='first')
            .fill_nan(0)
            .fill_null(0)
            .with_columns(pl.sum_horizontal(sources).alias('total'))
        )

        region_counts = df.select(sources).to_numpy()
        winner_idx = np.argmax(region_counts, axis=1).astype(int)
        df = df.with_columns(pl.Series([sources[idx] for idx in winner_idx]).alias('winner'))

        return df


class RoiSubregionDataFrame(DataFrameWrapper):
    """RoiSubregionDataFrame with each source per row, column shows the subregions"""

    _profile_required_fields = ('source', 'counts', 'total', 'total_fraction')

    def __init__(self, region: str, df: pl.DataFrame, profile: pl.DataFrame):
        """
        :param region: region name
        :param df: subregion dataframe
        :param profile: profile dataframe
        """
        self._region = region
        self._df = df
        self._profile = profile

        for field in self._profile_required_fields:
            if field not in profile.columns:
                raise RuntimeError(f'field not found: {field} -> {df.columns}')

    def __repr__(self):
        return repr(self.dataframe())

    def dataframe(self, dataframe: pl.DataFrame = None, may_inplace=True):
        """
        RoiSubregionDataFrame (VIS as example)::

            ┌─────────┬───────────┬───────────┬───────────┬───┬──────────┬──────────┬──────────┬──────────┐
            │ source  ┆ VISam     ┆ VISp      ┆ VISpm     ┆ … ┆ VISal    ┆ VISpor   ┆ VISli    ┆ VISpl    │
            │ ---     ┆ ---       ┆ ---       ┆ ---       ┆   ┆ ---      ┆ ---      ┆ ---      ┆ ---      │
            │ str     ┆ f64       ┆ f64       ┆ f64       ┆   ┆ f64      ┆ f64      ┆ f64      ┆ f64      │
            ╞═════════╪═══════════╪═══════════╪═══════════╪═══╪══════════╪══════════╪══════════╪══════════╡
            │ overlap ┆ 39.649682 ┆ 15.127389 ┆ 28.025478 ┆ … ┆ 3.025478 ┆ 2.707006 ┆ 1.592357 ┆ 0.159236 │
            │ aRSC    ┆ 32.160414 ┆ 28.952135 ┆ 23.05304  ┆ … ┆ 6.080207 ┆ 1.293661 ┆ 2.069858 ┆ 0.07762  │
            │ pRSC    ┆ 25.947955 ┆ 27.95539  ┆ 27.459727 ┆ … ┆ 3.122677 ┆ 2.973978 ┆ 1.982652 ┆ 1.016109 │
            └─────────┴───────────┴───────────┴───────────┴───┴──────────┴──────────┴──────────┴──────────┘
        """
        if dataframe is None:
            return self._df
        else:
            ret = RoiSubregionDataFrame(self._region, dataframe, self._profile)
            return ret

    @property
    def region(self) -> str:
        """region name"""
        return self._region

    @property
    def subregion(self) -> list[str]:
        """list of subregion names"""
        return self.drop('source').columns

    @property
    def n_subregion(self) -> int:
        """number of subregion"""
        return len(self.subregion)

    @property
    def profile(self) -> pl.DataFrame:
        """
        with channel-wise profile::

            ┌─────────┬────────┬───────┬────────────────┐
            │ source  ┆ counts ┆ total ┆ total_fraction │
            │ ---     ┆ ---    ┆ ---   ┆ ---            │
            │ str     ┆ u32    ┆ u32   ┆ f64            │
            ╞═════════╪════════╪═══════╪════════════════╡
            │ overlap ┆ 628    ┆ 4027  ┆ 0.155947       │
            │ aRSC    ┆ 3865   ┆ 30609 ┆ 0.12627        │
            │ pRSC    ┆ 4035   ┆ 14442 ┆ 0.279393       │
            └─────────┴────────┴───────┴────────────────┘
        """
        return self._profile

    @property
    def sources(self) -> list[str]:
        """list of source names"""
        return self['source'].to_list()

    def filter_overlap(self) -> Self:
        """filter out overlap source"""
        expr = pl.col('source') != 'overlap'
        self._profile = self._profile.filter(expr)
        return self.filter(expr)

    def with_animal_column(self, animal) -> Self:
        """with animal id column"""
        return self.with_columns(pl.lit(animal).alias('animal'))

    def to_dict(self, as_series: bool = True) -> dict[str, list[float]]:
        """to subregion:value dict"""
        return self.dataframe().select(pl.exclude('source')).to_dict(as_series=as_series)

    def to_numpy(self) -> np.ndarray:
        """to value array. `Array[float, [n_source, n_subregion]]`"""
        return self.dataframe().select(pl.exclude('source')).to_numpy()
