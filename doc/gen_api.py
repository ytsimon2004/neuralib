import importlib
import sys
import textwrap
from pathlib import Path

# === Configuration ===
SRC = Path('../src/neuralib').resolve()
DST = Path('source/api')
DST.mkdir(parents=True, exist_ok=True)

AUTOSUMMARY_DIR = DST / '_autosummary'
AUTOSUMMARY_DIR.mkdir(exist_ok=True)

CONTENT_FILE = """\
{module}
{underline}

.. currentmodule:: {module}

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

{autosummary_list}
"""

CONTENT_DIR = """\
{module}
{underline}

.. automodule:: {module}
   :members:

.. toctree::
    :maxdepth: 1
    :caption: Modules:

{module_list}
"""

# Track generated .rst files and autosummary targets
generated_rst_files = ['neuralib.rst']
autosummary_targets = []

# Add source root for import resolution
sys.path.insert(0, str(SRC.parent))


def get_module_all(module_path: Path) -> list:
    """Dynamically import a module and extract __all__, if available."""
    try:
        rel_path = module_path.relative_to(SRC.parent)
        modname = '.'.join(rel_path.with_suffix('').parts)
        mod = importlib.import_module(modname)
        return getattr(mod, '__all__', [])
    except Exception as e:
        print(f"[Warning] Could not extract __all__ from {module_path.name}: {e}")
        return []


def write_module_file(module: str, output_path: Path, all_list: list):
    """Write an .rst file for a module using only autosummary."""
    if not all_list:
        print(f"[Skipped] {module} has no __all__")
        return

    autosummary_list = textwrap.indent('\n'.join(all_list), '   ')
    content = CONTENT_FILE.format(
        module=module,
        underline='=' * len(module),
        autosummary_list=autosummary_list
    )
    autosummary_targets.extend([f"{module}.{name}" for name in all_list])
    output_path.write_text(content)
    print(f"[Created] {output_path}")


def write_directory_index(module: str, output_path: Path, module_list: list):
    """Write an .rst index file for a package directory."""
    module_list.sort()
    formatted_list = textwrap.indent('\n'.join(module_list), '    ')
    content = CONTENT_DIR.format(
        module=module,
        underline='=' * len(module),
        module_list=formatted_list
    )
    output_path.write_text(content)
    print(f"[Created] {output_path}")


def process_source_tree():
    """Walk through the source tree and generate .rst files."""
    for path in SRC.rglob('*'):
        rel = path.relative_to(SRC.parent)
        module_path = DST / (str(rel.with_suffix('.rst')).replace('/', '.'))

        if path.is_file() and path.suffix == '.py' and not path.name.startswith('_'):
            generated_rst_files.append(module_path.name)
            if not module_path.exists():
                modname = '.'.join(rel.with_suffix('').parts)
                all_list = get_module_all(path)
                write_module_file(modname, module_path, all_list)

        elif path.is_dir() and (path / '__init__.py').exists() and '__pycache__' not in path.parts:
            generated_rst_files.append(module_path.name)
            if not module_path.exists():
                modname = '.'.join(rel.parts)
                submodules = []
                for child in path.iterdir():
                    if child.suffix == '.py' and not child.name.startswith('_'):
                        submodules.append(f"{modname}.{child.stem}")
                    elif child.is_dir() and (child / '__init__.py').exists():
                        submodules.append(f"{modname}.{child.name}")
                write_directory_index(modname, module_path, submodules)


def cleanup_stale_rst():
    """Remove stale .rst files that are no longer needed."""
    for f in DST.glob('*.rst'):
        if f.name not in generated_rst_files:
            print(f"[Stale] {f.name}")
            # f.unlink()  # auto-delete


if __name__ == '__main__':
    process_source_tree()
    cleanup_stale_rst()
