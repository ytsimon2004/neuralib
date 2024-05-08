import contextlib
from io import BytesIO
from typing import ContextManager

import requests
from tqdm import tqdm

__all__ = ['download_with_tqdm',
           'tqdm_joblib']


def download_with_tqdm(url: str) -> BytesIO:
    """download url with tqdm bar
    Use in large file downloading,

    :param url: download URL
    :return: BytesIO
    """
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    file_size = int(resp.headers.get('content-length', 0))
    progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True)
    content_stream = BytesIO()

    for data in resp.iter_content(chunk_size=1024):
        content_stream.write(data)
        progress_bar.update(len(data))

    progress_bar.close()
    content_stream.seek(0)

    return content_stream


@contextlib.contextmanager
def tqdm_joblib(tqdm_obj: tqdm) -> ContextManager[tqdm]:
    """Context manager to patch joblib multiprocessing to report into tqdm progress bar given as argument

    Example of running foreach neuron shuffle:

    .. code-block:: python

        from joblib import Parallel, delayed

        func = ... # Callable[*args, Any], arg include foreach *n*

        with tqdm_joblib(tqdm(desc="lower bound", unit='neuron', ncols=80)) as _:
            Parallel(n_jobs=self.parallel_jobs, backend='multiprocessing', verbose=True)(
                delayed(func)
                (*args)
                for n in neuron_list

    :param tqdm_obj: tqdm object
    :return: tqdm context manger

    """
    import joblib

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_obj.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield tqdm_obj
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_obj.close()
