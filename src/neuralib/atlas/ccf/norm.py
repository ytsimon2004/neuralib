from __future__ import annotations

from typing import Literal, get_args, Iterable

import polars as pl

from neuralib.util.util_verbose import fprint

__all__ = [
    'ROIS_NORM_TYPE',
    'MouseBrainRoiNormHandler',
    'handle_failure_norm',
    'foreach_norm_method'
]

ROIS_NORM_TYPE = Literal['channel', 'volume', 'cell']


class MouseBrainRoiNormHandler:
    """Class for handle the normalization for ROIs across mouse brain"""

    def __init__(self, norm_type: ROIS_NORM_TYPE | None = None):
        """
        :param norm_type: which kind of normalization for the roi labeling

            *``channel``: normalize to fraction of rois for a specific color fluorescence channel

            *``volume``: normalize to the volume size per region (cellatlas-based) # TODO validate with allenSDK source

            *``cell``: normalize to the total cell counts per region (cellatlas-based)
        """
        if norm_type is not None and norm_type not in get_args(ROIS_NORM_TYPE):
            raise ValueError('')

        self.norm_type = norm_type

    @classmethod
    def handle_failure(cls, df: pl.DataFrame, *,
                       expr: pl.Expr | None = None,
                       drop_not_found: bool = True) -> pl.DataFrame:
        return handle_failure_norm(df, expr=expr, drop_not_found=drop_not_found)

    @property
    def unit_label(self) -> str:
        """For plotting"""
        if self.norm_type is None:
            return 'Counts#'
        elif self.norm_type == 'cell':
            return 'Cell density(%)'
        elif self.norm_type == 'volume':
            return 'Density(cells-mm^3)'
        elif self.norm_type == 'channel':
            return 'Fraction of input(%)'
        else:
            raise ValueError('')

    @property
    def unit_col(self) -> str:
        """For dataframe column"""
        if self.norm_type is None:
            return 'n_rois'
        elif self.norm_type == 'channel':
            return 'percent'
        elif self.norm_type in ('volume', 'cell'):
            return f'{self.norm_type}_norm_n_rois'
        else:
            raise ValueError('')

    def expand(self,
               df: pl.DataFrame,
               area_col: str,
               expand_cols: list[str]) -> pl.DataFrame:
        """
        expand the dataframe based on different normalization

        :param df: input dataframe has abs cell counts per region
        :param area_col: brain region acronym for querying
        :param expand_cols: columns with abs cell counts need to be expanded
        :return: expanded dataframe
        """
        if self.norm_type in ('cell', 'volume'):
            ret = self._compute_cellatlas_expand_field(df, area_col, expand_cols)
        elif self.norm_type == 'channel':  # calculate in RoiClassifier.percent_sorted_df
            ret = df
        else:
            raise ValueError(f'{self.norm_type} unknown')

        return ret

    def _compute_cellatlas_expand_field(self, df, area_col, expand_cols) -> pl.DataFrame:
        from neuralib.atlas.cellatlas import CellAtlas
        ctlas = CellAtlas.load_sync_allen_structure_tree()

        if self.norm_type == 'cell':
            calc = ((pl.col(c) / pl.col('n_neurons') * 100).alias(f'cell_norm_{c}') for c in expand_cols)
        elif self.norm_type == 'volume':
            calc = ((pl.col(c) / pl.col('Volumes [mm^3]')).alias(f'volume_norm_{c}') for c in expand_cols)

        sync = (
            df.join(ctlas.rename({'acronym': area_col}), on=area_col, how='left')
            .with_columns(calc)
            .fill_null(-1)
            .fill_nan(-1)
        )

        return sync


def handle_failure_norm(df: pl.DataFrame, *,
                        expr: pl.Expr | None = None,
                        drop_not_found: bool = True):
    """
    warning verbose or drop if
    cellatlas cannot find the n_neurons OR volume in the given acronym,

    :param df:
    :param expr: condition for determine if it's failure
    :param drop_not_found: whether drop the region that can not be found
    """
    if expr is None:
        expr = pl.col('name').is_null()

    failure = df.filter(expr)
    if failure.shape[0] == 0:
        return df

    msg = '\n#===== WARNING =====#'
    msg += f'\n{failure}'
    msg += '\nIF IMPORTANT INFO LOSS, please decrease the merge level ' \
           'or check details in cell atlas APP/csv & allenccf raw csv!!!'

    if drop_not_found:
        msg += '\n#===== DROP AREAS ======#'
        df = df.filter(~expr)

    fprint(msg, vtype='warning')

    return df


def foreach_norm_method(include_abs_count: bool = True, **kwargs) -> Iterable[MouseBrainRoiNormHandler]:
    """
    foreach normalization method

    :param include_abs_count: if iterator include abs cell counts
    :param kwargs: pass to ``MouseBrainRoiNormHandler``
    :return:
    """
    methods = []
    if include_abs_count:
        methods = [None]
    methods += get_args(ROIS_NORM_TYPE)

    for t in methods:
        yield MouseBrainRoiNormHandler(t, **kwargs)
