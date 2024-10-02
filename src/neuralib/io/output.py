from pathlib import Path

from neuralib.typing import PathLike

__all__ = ['mkdir_version']


def mkdir_version(root: PathLike,
                  folder_name: str,
                  reuse_latest: bool = False) -> Path:
    """
    Make directory with version number, and auto-accumulate

    :param root: Root directory where the versioned folder will be created.
    :param folder_name: Name of the folder for which versions will be created. Can contain sub-paths.
    :param reuse_latest: Boolean flag indicating whether to reuse the latest version folder if it exists.
    :return: Path of the created or reused version folder.
    """
    root = Path(root)

    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    if '/' in folder_name:
        i = folder_name.rindex('/')
        root = root / folder_name[:i]
        folder_name = folder_name[i + 1:]

    output = root / f'{folder_name}_v0'
    end = len(output.name) - 1
    versions = list(root.glob(f'{folder_name}_v*'))
    n_versions = len(versions)

    if output.exists() or n_versions != 0:  # either has v0 or other versions
        v = 1

        if n_versions != 0:
            max_version = max([int(d.name[end:]) for d in versions])
            v = max_version + 1 if not reuse_latest else max_version

        output = root / f'{folder_name}_v{v}'

    output.mkdir(parents=True, exist_ok=True)

    return output
