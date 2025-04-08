import tempfile
from pathlib import Path

from neuralib.io import mkdir_version


def test_mkdir_version():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        path1 = mkdir_version(root, 'dir')
        path2 = mkdir_version(root, 'dir')
        path3 = mkdir_version(root, 'dir')
        path4 = mkdir_version(root, 'dir', reuse_latest=True)

        assert path1.name == 'dir_v0'
        assert path2.name == 'dir_v1'
        assert path3.name == 'dir_v2'
        assert path4.name == 'dir_v2'
