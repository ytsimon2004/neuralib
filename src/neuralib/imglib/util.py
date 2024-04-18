from __future__ import annotations

import cv2
import imageio.v2
import numpy as np
from tqdm import tqdm

from neuralib.util.util_type import PathLike

__all__ = [
    #
    'read_sequences',
    'write_avi',
    'read_pdf',
    #
    'tif_to_gif',
    'normalize_sequences',
    'handle_invalid_value'
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

    for frame in tqdm(frames):
        # frame = frame[:, :, [2, 1, 0]]  # Convert RGB to BGR
        out.write(frame)

    out.release()


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
    frames = imageio.mimread(image_file)
    frames = normalize_sequences(frames, **kwargs)
    imageio.mimsave(output_path, frames, fps=fps)


def normalize_sequences(frames: list[np.ndarray] | np.ndarray,
                        handle_invalid: bool = True,
                        gamma_correction: bool = False,
                        gamma_value: float = 0.5,
                        to_8bit: bool = False) -> list[np.ndarray]:
    """
    Do the normalization for the image sequences

    :param frames: list of image array
    :param handle_invalid: handle Nan and negative value
    :param gamma_correction: to the gamma correction
    :param gamma_value: gamma correction value
    :param to_8bit: to 8bit images
    :return: list of normalized image array
    """
    if handle_invalid:
        frames = handle_invalid_value(frames)

    if gamma_correction:
        frames = [np.power(f, gamma_value) for f in frames]

    # find global min and max across all frames
    global_min = min(frame.min() for frame in frames)
    global_max = max(frame.max() for frame in frames)

    # normalize and scale to 0-255
    ret = [(frame - global_min) / (global_max - global_min) * 255 for frame in frames]

    if to_8bit:
        ret = [frame.astype('uint8') for frame in ret]

    return ret


def handle_invalid_value(frames: list[np.ndarray] | np.ndarray) -> list[np.ndarray]:
    """Handle NaN and negative values and ensure all values are >= 0"""
    frames = [np.nan_to_num(frame, nan=0, posinf=0, neginf=0) for frame in frames]
    frames = [np.clip(frame, 0, None) for frame in frames]
    return frames
