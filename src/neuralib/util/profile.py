
import functools
import sys
import threading
import time

from neuralib.util.verbose import fprint

__all__ = [
    'profile_test',
    'trace_line'
]


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


def trace_line(f=None, *, interval: float = 0.1):
    """
    Decorator for trace line number in real time using threading.

    **Use Case**

    - OS directly kill job without traceback

    :param f: Decorated function
    :param interval: Time interval for trace (in second).
    """

    def _decorator(f):

        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            done = False
            _file = None
            _lineno = None
            _prev_time = None

            def update():
                nonlocal done
                nonlocal _file
                nonlocal _lineno
                nonlocal _prev_time

                while not done:
                    time.sleep(interval)

                    # noinspection PyUnresolvedReferences,PyProtectedMember
                    info = sys._current_frames()[caller_thread]

                    if info.f_code.co_filename != _file or info.f_lineno != _lineno:

                        current_time = time.time()
                        if _prev_time is not None:
                            elapsed_time = current_time - _prev_time
                        else:
                            elapsed_time = 0  # First iteration

                        _file = info.f_code.co_filename
                        _lineno = info.f_lineno
                        _prev_time = current_time

                        fprint(f'File: {_file} in line {_lineno} - Time elapsed: {elapsed_time:.3f} sec',
                               vtype='debug',
                               flush=True)

            thread = threading.Thread(target=update)
            caller_thread = threading.current_thread().ident
            thread.start()

            try:
                result = f(*args, **kwargs)
            finally:
                done = True  # Signal to stop the memory monitoring thread
                thread.join()  # Ensure the thread finishes

            return result

        return _wrapper

    if f is None:
        return _decorator
    else:
        return _decorator(f)
