from __future__ import annotations

import cv2
import imageio.v2
import numpy as np

from neuralib.util.util_type import PathLike

__all__ = [
    #
    'read_avi',
    'pdf_as_image',
    #
    'tif_to_gif',
    'normalize_sequences',
    'handle_invalid_value'
]


def read_avi(avi_file: PathLike,
             grey_scale: bool = True) -> np.ndarray:
    """simple read for avi file, and collect as image array (F, W, H, *3). *optional"""
    cap = cv2.VideoCapture(str(avi_file))
    if not cap.isOpened():
        raise RuntimeError(f'error opening avi: {avi_file}')

    ret = []
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break

        if grey_scale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret.append(frame)

    return np.array(ret)


def pdf_as_image(file: PathLike,
                 single_image: bool = True,
                 poppler_path: str | None = None,
                 **kwargs) -> np.ndarray:
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFInfoNotInstalledError

    if single_image:
        try:
            conv = convert_from_path(file, **kwargs)[0]
        except PDFInfoNotInstalledError:
            if poppler_path is None:
                # https://stackoverflow.com/questions/53481088/poppler-in-path-for-pdf2image
                poppler_path = r'C:\Users\yu-ting\Downloads\Release-23.08.0-0\poppler-23.08.0\Library\bin'
            conv = convert_from_path(file, poppler_path=poppler_path, **kwargs)[0]
    else:
        raise NotImplementedError('')

    return np.array(conv)


def tif_to_gif(image_file: PathLike,
               output_path: PathLike,
               fps: int = 30,
               **kwargs) -> None:
    """convert tif sequences to GIF"""
    frames = imageio.mimread(image_file)
    frames = normalize_sequences(frames, **kwargs)
    imageio.mimsave(output_path, frames, fps=fps)


def normalize_sequences(frames: list[np.ndarray],
                        handle_invalid: bool = True,
                        gamma_correction: bool = False,
                        gamma_value: float = 0.5,
                        to_8bit: bool = False) -> list[np.ndarray]:
    """
    Do the normalization for the image sequences

    :param frames: list of image array
    :param handle_invalid: handle Nan and negative value
    :param gamma_correction:
    :param gamma_value
    :param to_8bit
    :return:
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


def handle_invalid_value(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Handle NaN and negative values and ensure all values are >= 0"""
    frames = [np.nan_to_num(frame, nan=0, posinf=0, neginf=0) for frame in frames]
    frames = [np.clip(frame, 0, None) for frame in frames]
    return frames
