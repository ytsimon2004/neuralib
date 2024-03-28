import os
import subprocess

__all__ = ['NERUOLIB_VERSION',
           'get_commit_hash']


def get_commit_hash() -> str:
    codedir = os.path.dirname(os.path.abspath(__file__))
    try:
        commit_ref = (
            subprocess
            .check_output(['git', 'rev-parse',
                           '--verify', 'HEAD',
                           '--short'], cwd=codedir)
            .decode()
            .strip('\n')
        )

    except subprocess.CalledProcessError:
        commit_ref = ''

    return commit_ref


NERUOLIB_VERSION = get_commit_hash()
