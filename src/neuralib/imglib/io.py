from __future__ import annotations

import cv2
import numpy as np
from tqdm import tqdm

from neuralib.imglib.norm import normalize_sequences
from neuralib.util.util_type import PathLike

__all__ = [
    'read_sequences',
    'read_pdf',
    'write_avi',
    'tif_to_gif'
]


def read_sequences(avi_file: str | PathLike,
                   grey_scale: bool = True) -> np.ndarray:
    """Read an sequences file (i.e., AVI/MP4) into an array, converting to grayscale if specified.

    :param avi_file: Path to the AVI file.
    :param grey_scale: Whether to convert frames to grayscale.
    :return: (F, W, H, <3>) sequences array
    """
    cap = cv2.VideoCapture(str(avi_file))
    if not cap.isOpened():
        raise RuntimeError(f'Error opening AVI file: {avi_file}')

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if grey_scale:
        shape = (frame_count, height, width)
    else:
        shape = (frame_count, height, width, 3)

    frames = np.empty(shape, dtype=np.uint8)
    i = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if grey_scale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames[i] = frame
            i += 1
    finally:
        cap.release()

    return frames


def read_pdf(file: PathLike,
             single_image: bool = True,
             poppler_path: str | None = None,
             **kwargs) -> np.ndarray:
    """
    Read pdf as an image array

    :param file:
    :param single_image:
    :param poppler_path: if poppler not installed in a system level. Specify the path.
        for example: ``Release-23.08.0-0\poppler-23.08.0\Library\bin``

        .. seealso::

            `<https://stackoverflow.com/questions/53481088/poppler-in-path-for-pdf2image>`_
    :param kwargs: pass through `convert_from_path`
    :return: image array
    """
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFInfoNotInstalledError

    if single_image:
        try:
            conv = convert_from_path(file, **kwargs)[0]
        except PDFInfoNotInstalledError:
            if poppler_path is None:
                raise RuntimeError('download poppler first. or using apt-get / brew depending on OS')

            conv = convert_from_path(file, poppler_path=poppler_path, **kwargs)[0]
    else:
        raise NotImplementedError('multi-pages pdf not currently support')

    return cv2.cvtColor(np.array(conv), cv2.COLOR_BGR2RGB)


def tif_to_gif(image_file: PathLike,
               output_path: PathLike,
               fps: int = 30,
               **kwargs) -> None:
    """convert tif sequences to GIF"""
    import imageio

    frames = imageio.mimread(image_file)
    frames = normalize_sequences(frames, **kwargs)
    imageio.mimsave(output_path, frames, fps=fps)


def write_avi(file_path: str, frames: np.ndarray, fps: int = 30.0):
    """
    Write a sequence of frames to an AVI file.

    :param file_path: The path where the AVI file will be saved.
    :param frames: A NumPy array of shape (F, H, W, 3) containing the frames.
    :param fps: The frames per second (frame rate) of the output video.
    """
    if not file_path.endswith('avi'):
        raise ValueError('file path need to write to *.avi')

    if not len(frames):
        raise ValueError("Frames array is empty.")

    height, width = frames[0].shape[:2]

    if frames[0].ndim == 3:
        is_color = True
    else:
        is_color = False

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height), isColor=is_color)

    for frame in tqdm(frames, desc='write frames to avi', unit='frames'):
        # frame = frame[:, :, [2, 1, 0]]  # Convert RGB to BGR
        out.write(frame)

    out.release()
