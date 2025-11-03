import cv2
import numpy as np

__all__ = [
    'normalize_sequences',
    'handle_invalid_value',
    'get_percentile_value',
    'enhance_blood_vessels'
]


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


def get_percentile_value(im: np.ndarray,
                         perc_interval: tuple[float, float] = (10, 100)) -> tuple[float, float]:
    """Get the central distribution boundary value for
    imaging enhancement by changing the scaling of array.

    :param im: image array
    :param perc_interval: percentile
    :return: lower_bound and upper_bound based on value distribution
    """
    im = im.flatten()
    lb, up = perc_interval
    return np.percentile(im, lb), np.percentile(im, up)


def enhance_blood_vessels(image: np.ndarray, enhance_contrast: bool = True) -> np.ndarray:
    """
    Enhance blood vessel patterns in the image.

    :param image: Input image (grayscale or color)
    :param enhance_contrast: Whether to apply CLAHE for contrast enhancement
    :return: Enhanced grayscale image with blood vessels highlighted
    """
    # grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # normalize to 0-255 range
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # apply CLAHE for contrast enhancement
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
    else:
        enhanced = gray

    # apply bilateral filter to preserve edges while reducing noise
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

    return filtered
