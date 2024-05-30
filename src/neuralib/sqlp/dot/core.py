import subprocess
import sys
from pathlib import Path
from typing import overload, IO, Any, Union

from .. import Database
from ..table import _table_class, table_name, Table

__all__ = ['generate_dot']


@overload
def generate_dot(db: Database) -> str:
    pass


@overload
def generate_dot(db: Database, file: Union[str, Path]) -> None:
    pass


@overload
def generate_dot(db: Database, file: IO) -> None:
    pass


def generate_dot(db: Database, file=None):
    """

    :param db:
    :param file: io, or an output file path which support '.puml' file and '.png' file (required ``graphivz``).
    :return:
    """
    if file is None:
        from io import StringIO
        buf = StringIO()
        _generate_dot(db, buf)
        return buf.getvalue()
    elif isinstance(file, (str, Path)):
        file = Path(file)
        if file.suffix in ('.pu', '.puml'):
            with file.open() as out:
                _generate_dot(db, out)
        elif file.suffix == '.png':
            _generate_dot_png(generate_dot(db), file)
        else:
            raise RuntimeError(f'unsupported filetype : {file.suffix}')
    else:
        _generate_dot(db, file)


def _generate_dot_png(dot: str, file: Path) -> int:
    p = subprocess.Popen(['dot', '-Tpng', '-o', str(file)], stdin=subprocess.PIPE)
    p.communicate(dot.encode())
    return p.wait()


def _generate_dot(db: Database, file=sys.stdout, *,
                  graph: dict[str, Any] = None,
                  node: dict[str, Any] = None,
                  edge: dict[str, Any] = None):
    name = db.database_file
    if name is None:
        name = ':memory:'
    print('digraph', f'"{name}"', '{', file=file)

    _graph = dict()
    if graph is not None:
        _graph.update(graph)

    if len(_graph):
        print('graph [', file=file)
        for k, v in _graph.items():
            print(k, '=', v, file=file)
        print('];', file=file)

    _node = dict(shape='record', rankdir='LR')
    if node is not None:
        _node.update(node)

    if len(_node):
        print('node [', file=file)
        for k, v in _node.items():
            print(k, '=', v, file=file)
        print('];', file=file)

    _edge = dict()
    if edge is not None:
        _edge.update(edge)

    if len(_edge):
        print('edge [', file=file)
        for k, v in _edge.items():
            print(k, '=', v, file=file)
        print('];', file=file)

    _generate_dot_table(db, file)
    print('}', file=file)


def _generate_dot_table(db: Database, file=sys.stdout):
    for table in db.database_tables:
        table: Table = _table_class(table)
        print(f'"{table.table_name}"', '[', file=file)

        pf = table.table_primary_field_names
        uf = table.table_unique_field_names
        ff = []

        for field in table.table_fields:
            at = []
            if field.name in pf:
                at.append('#')
            elif field.name in uf:
                at.append('!')
            at.append(field.name)
            at.append(':')
            at.append(field.f_type.__name__)
            if not field.not_null:
                at.append('?')
            ff.append(''.join(at))

        print('label=', '"', '<name>', table.table_name, '|{', '|'.join(ff), '}"', sep='', file=file)

        print('];', file=file)

    for table in db.database_tables:
        table: Table = _table_class(table)
        for foreign in table.table_foreign_fields:
            if foreign.fields == foreign.foreign_fields:
                print(table_name(foreign.table), ':name', '->', table_name(foreign.foreign_table), ':name', file=file)
            else:
                k1 = ', '.join(foreign.fields)
                k2 = ', '.join(foreign.foreign_fields)

                print(table_name(foreign.table), ':name', '->', table_name(foreign.foreign_table), ':name', '[', file=file)
                print('label=', '"(', k1, ') to (', k2, ')"', file=file)
                print('];', file=file)
