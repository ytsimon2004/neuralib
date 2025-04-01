from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import polars as pl
from polars.polars import ColumnNotFoundError
from typing_extensions import Self

from neuralib.atlas.cellatlas import load_cellatlas
from neuralib.atlas.data import load_bg_volumes
from neuralib.atlas.map import merge_until_level, NUM_MERGE_LAYER, DEFAULT_FAMILY_DICT
from neuralib.atlas.typing import Channel, HEMISPHERE_TYPE
from neuralib.typing import PathLike
from neuralib.util.dataframe import DataFrameWrapper
from neuralib.util.verbose import print_save

__all__ = [
    'ROIS_NORM_TYPE',
    'RoiClassifierDataFrame',
    'RoiNormalizedDataFrame'
]

ROIS_NORM_TYPE = Literal['channel', 'volume', 'cell', 'none']


class RoiClassifierDataFrame(DataFrameWrapper):
    _required_fields = ('acronym', 'AP_location', 'DV_location', 'ML_location', 'channel', 'source')
    _valid_classified_column = ('acronym', 'tree_0', 'tree_1', 'tree_2', 'tree_3', 'tree_4')

    def __init__(self, df: pl.DataFrame,
                 cached_dir: PathLike | None = None):
        self._df = df
        self._cached_dir = cached_dir

        for field in self._required_fields:
            if field not in df.columns:
                raise RuntimeError(f'field not found: {field} -> {df.columns}')

    def __repr__(self):
        return repr(self.dataframe())

    def dataframe(self, dataframe: pl.DataFrame = None, may_inplace=True):
        if dataframe is None:
            return self._df
        else:
            ret = RoiClassifierDataFrame(dataframe)
            return ret

    @property
    def channels(self) -> list[Channel]:
        return self.dataframe()['channels'].unique().to_list()

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def channel_counts(self) -> pl.DataFrame:
        return self['source'].value_counts()

    @property
    def sources(self) -> list[Channel]:
        return self.dataframe()['source'].unique().to_list()

    @property
    def n_sources(self) -> int:
        return len(self.sources)

    @property
    def is_overlapped_channel(self) -> bool:
        return 'overlap' in self.channels

    def get_channel_source_dict(self) -> dict[str, str]:
        return {
            it[0]: it[1]
            for it in self['channel', 'source'].unique().iter_rows()
        }

    def get_classified_column(self, level: int | None) -> str:
        match level:
            case None:
                col = 'acronym'
            case int():
                col = f'tree_{level}'
            case _:
                raise TypeError('')

        if col not in self._valid_classified_column:
            raise ValueError(f'invalid level: {level}')
        if col not in self.dataframe().columns:
            raise ColumnNotFoundError(f'{col} not found: {self.dataframe().columns}')

        return col

    def post_processing(self, *,
                        filter_capital: bool = True,
                        tree: bool = True,
                        family: bool = True,
                        hemisphere: bool = True,
                        copy_overlap: bool = True,
                        filter_injection: tuple[str, str] | None = None,
                        force: bool = False) -> Self:
        """

        :param filter_capital:
        :param tree:
        :param family:
        :param hemisphere:
        :param copy_overlap:
        :param filter_injection:
        :param force:
        :return:
        """
        if self._cached_dir is not None:
            file = Path(self._cached_dir) / 'parsed_roi.csv'

            if file.exists() and not force:
                return RoiClassifierDataFrame(pl.read_csv(file), self._cached_dir)
            else:
                ret = self._post_processing(filter_capital, tree, family, hemisphere, copy_overlap, filter_injection)
                ret.dataframe().write_csv(file)
                print_save(file)
        else:
            ret = self._post_processing(filter_capital, tree, family, hemisphere, copy_overlap, filter_injection)

        return ret

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
        if filter_injection is not None:
            ret = ret.filter_injection_site(area=filter_injection[0], hemisphere=filter_injection[1])

        return ret

    def filter_injection_site(self, area: str, hemisphere: str) -> Self:
        expr1 = pl.col('acronym').str.starts_with(area)
        expr2 = pl.col('hemisphere') == hemisphere
        return self.filter(~(expr1 & expr2))

    def filter_capital_name(self) -> Self:
        return self.filter(pl.col('acronym').str.contains(r'[A-Z]+'))

    def with_tree_columns(self) -> Self:
        acronym = self['acronym']
        return self.with_columns(
            pl.Series(name=f'tree_{level}', values=merge_until_level(acronym, level))
            for level in range(NUM_MERGE_LAYER)
        )

    def with_family_columns(self) -> Self:
        def get_family(row) -> str:
            for name, family in DEFAULT_FAMILY_DICT.items():
                if row in family:
                    return name
            return 'unknown'

        return self.with_columns(pl.col('tree_0').map_elements(get_family, return_dtype=pl.Utf8).alias('family'))

    def with_hemisphere_column(self, invert: bool = False) -> Self:
        if invert:
            expr = pl.when(pl.col('ML_location') < 0)
        else:
            expr = pl.when(pl.col('ML_location') >= 0)

        return self.with_columns(expr.then(pl.lit('ipsi')).otherwise(pl.lit('contra')).alias('hemisphere'))

    def with_overlap_copy(self) -> Self:
        """Copy overlap channels counts to individual channels"""
        ret = [self.dataframe()]
        for channel, source in self.get_channel_source_dict().items():
            if channel not in 'overlap':
                df = (
                    self.dataframe().filter(pl.col('channel') == 'overlap')
                    .with_columns(pl.lit(channel).alias('channel'))
                    .with_columns(pl.lit(source).alias('source'))
                )
                ret.append(df)

        return RoiClassifierDataFrame(pl.concat(ret), self._cached_dir)

    # ==================== #
    # Normalized DataFrame #
    # ==================== #

    def to_normalized(self, norm: ROIS_NORM_TYPE,
                      level: int | None, *,
                      source: str | None = None,
                      top_area: int | None = None,
                      rest_as_others: bool = False,
                      hemisphere: HEMISPHERE_TYPE = 'both',
                      animal: str | None = None,
                      volume_norm_backend: Literal['cellatlas', 'brainglobe'] = 'cellatlas') -> RoiNormalizedDataFrame:
        """

        :param level:
        :param norm:
        :param source:
        :param top_area:
        :param rest_as_others: vertical concat a region called **other** with the roi filtered by ``top area``
        :param hemisphere:
        :param animal:
        :param volume_norm_backend:
        :return:
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
            total_counts = self.channel_counts.filter(pl.col('source') == source)['count'].item()
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


# TODO maybe do not need NormHandler
class RoiNormalizedDataFrame(DataFrameWrapper):
    _required_fields = ('counts', 'fraction', 'hemisphere')

    def __init__(self, df: pl.DataFrame,
                 classified_column: str,
                 normalized: ROIS_NORM_TYPE):

        for field in self._required_fields:
            if field not in df.columns:
                raise RuntimeError(f'field not found: {field}')

        self._df = df
        self._classified_column = classified_column
        self._normalized = normalized

    @property
    def classified_column(self) -> str:
        """region classified column name"""
        return self._classified_column

    @property
    def value_column(self) -> str:
        match self._normalized:
            case 'volume' | 'cell' | 'channel':
                return 'normalized'
            case 'none':
                return 'counts'
            case _:
                raise ValueError(f'invalid normalized method: {self._normalized}')

    @property
    def normalized_unit(self) -> str:
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
        if isinstance(areas, str):
            areas = [areas]

        ret = self.filter(pl.col(self.classified_column).is_in(areas))
        if ret.dataframe().is_empty():
            raise ValueError(f'{areas} not found')

        return ret

    def to_bias_index(self, source_a, source_b) -> pl.DataFrame:
        """

        :param source_a:
        :param source_b:
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


# DIR move to ?

def _copy_coronal():
    pass


def _copy_sagittal():
    pass
