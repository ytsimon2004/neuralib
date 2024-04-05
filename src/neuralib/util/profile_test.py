import functools
import os

import psutil

from neuralib.util.util_verbose import fprint

__all__ = ['profile_test']


class profile_test:
    def __init__(self, enable=False, output_file='profile.png'):
        self._enable = enable
        self._profile = None
        self._output_file = output_file

    def __enter__(self):
        if self._enable:
            import cProfile

            self._profile = cProfile.Profile()
            self._profile.enable()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._enable:
            self._profile.disable()

            if self._output_file.endswith('.dat'):
                self._profile.dump_stats(self._output_file)

            elif self._output_file.endswith('.txt'):
                import pstats
                stat = pstats.Stats(self._profile)
                with open(self._output_file, 'w') as f:
                    stat.stream = f
                    stat.print_stats()

            elif self._output_file.endswith('.png'):
                import subprocess

                f1 = self._output_file.replace('.png', '.dat')

                self._profile.dump_stats(f1)

                cmd_line = f'python -m gprof2dot -f pstats {f1} | dot -T png -o {self._output_file}'

                with subprocess.Popen(['bash', '-c', cmd_line]) as proc:
                    proc.wait()


def process_memory() -> int:
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def mem_profile():
    """TODO compare with memory_profiler.profile decorator"""

    def _decorator(f):
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            before = process_memory()
            f(*args, **kwargs)
            after = process_memory()
            fprint(f'func: <{f.__name__}> consumed memory {after - before}')

        return _wrapper

    return _decorator
