import requests
from tqdm import tqdm

__all__ = ['download_with_tqdm']


def download_with_tqdm(url: str) -> requests.Response:
    """download url with tqdm bar
    Use in large file downloading,
    """
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    file_size = int(resp.headers.get('content-length', 0))
    progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True)

    for data in resp.iter_content(chunk_size=1024):
        progress_bar.update(len(data))

    return resp
