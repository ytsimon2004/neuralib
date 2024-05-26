from __future__ import annotations

import functools
import inspect

from neuralib.util.func import create_fn
from . import expr

__all__ = ['as_func_expr']


def as_func_expr(f=None, *, func=expr.SqlFuncOper):
    # noinspection PyShadowingNames
    def _as_func_expr(f):
        func_name = f.__name__.upper()
        s = inspect.signature(f)
        para = []

        n_none = 0

        for n, p in s.parameters.items():
            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                para.append('*' + n)
                n_none = -1
                break
            elif p.default is inspect.Parameter.empty:
                para.append(n)
            elif p.default is None:
                para.append((n, 'None'))
                n_none += 1
            else:
                raise RuntimeError()

        if n_none == -1:
            args = ', '.join(para)
            code = [f'return SqlFuncOper("{func_name}", {args})']
        elif n_none == 0:
            args = ', '.join(para)
            code = [f'return SqlFuncOper("{func_name}", {args})']
        elif n_none == 1:
            assert isinstance(para[-1], tuple)
            code = []
            z = para[-1][0]
            code.append(f'if {z} is None:')
            args = ', '.join(para[:-1])
            code.append(f'  return SqlFuncOper("{func_name}", {args})')
            code.append('else:')
            args = args + ', ' + z
            code.append(f'  return SqlFuncOper("{func_name}", {args})')
        else:
            code = []
            args = ', '.join(para[:-n_none])
            for i in range(n_none, 0, -1):
                assert isinstance(para[-i], tuple)
                z = para[-i][0]
                _if = 'if' if len(code) == 0 else 'elif'
                code.append(f'{_if} {z} is None:')
                code.append(f'  return SqlFuncOper("{func_name}", {args})')
                args = args + ', ' + z
            code.append('else:')
            code.append(f'  return SqlFuncOper("{func_name}", {args})')

        ret = create_fn(f.__name__, para, '\n'.join(code), locals=dict(SqlFuncOper=func))
        return functools.wraps(f)(ret)

    if f is None:
        return _as_func_expr
    else:
        return _as_func_expr(f)
