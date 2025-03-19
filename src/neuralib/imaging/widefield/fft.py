import numpy as np

__all__ = ['SequenceFFT']


class SequenceFFT:

    def __init__(self, seq: np.ndarray,
                 component: int = 1):
        """
        :param seq: Image Sequence. `Array[float, [F, H, W]]`
        :param component: Which Fourier component (frequency) to use, defaults to 1 (primary oscillatory component)
        """
        self.seq = seq
        self._component = component

        # cache
        self._freq_map: np.ndarray | None = None
        self._intensity: np.ndarray | None = None

    @property
    def component(self) -> int:
        return self._component

    @property
    def n_frames(self) -> int:
        return self.seq.shape[0]

    @property
    def height(self) -> int:
        return self.seq.shape[1]

    @property
    def width(self) -> int:
        return self.seq.shape[2]

    def get_freq_map(self) -> np.ndarray:
        """
        Takes in trial-averaged sequence and calculates the Fourier Transform,
        basically transforming each pixel’s intensity over time (across frames)

        :return: `Array[float, [H, W]]`
        """
        if self._freq_map is None:
            fft_movie = np.fft.fft(self.seq, axis=0)
            freq = fft_movie[self.component]
            self._freq_map = freq
        return self._freq_map

    def get_intensity(self) -> np.ndarray:
        """Computes the magnitude of the frequency component.
        i.e., Strength of the responsiveness (frame-wise averaged)"""
        if self._intensity is None:
            self._intensity = np.abs(self.get_freq_map()) / self.n_frames
        return self._intensity

    def get_phase(self) -> np.ndarray:
        """Computes the phase of the selected frequency component.
        For example, the spatial locations of the visual field.

        :return: `Array[float, [H, W]]`
        """
        return -1 * np.angle(self.get_freq_map()) % (2 * np.pi)

    def as_colormap(self, *,
                    saturation_factor: float = 0.3,
                    value_perc: float = 98,
                    saturation_perc: float = 90,
                    to_rgb: bool = True) -> np.ndarray:
        """
        Generates an HSV colormap representation of the Fourier transform results.

        - Hue (H) represents the phase of the oscillation.
        - Saturation (S) represents the intensity raised to a power (`saturation_factor`).
        - Value (V) represents the normalized intensity of the frequency component.

        The generated colormap provides a visualization of frequency components
        across spatial locations.

        :param saturation_factor: Exponent applied to the intensity for controlling saturation scaling
        :param value_perc: Percentile threshold to normalize intensity values
        :param saturation_perc: Percentile threshold to normalize saturation values
        :param to_rgb: If True, converts the HSV colormap to RGB using OpenCV
        :return:
        """
        intensity = self.get_intensity()

        h = self.get_phase() / (2 * np.pi)  # mapping value from [0, 2pi] to [0, 1]
        s = intensity ** saturation_factor
        v = intensity

        # small value to black / white
        v /= np.percentile(intensity, value_perc)
        s /= np.percentile(s, saturation_perc)

        color_map = np.stack([h, s, v], axis=2).astype(np.float32)

        if to_rgb:
            import cv2
            color_map = np.clip(color_map, 0, 1) * 255
            color_map = cv2.cvtColor(color_map.astype(np.uint8), cv2.COLOR_HSV2RGB_FULL)

        return color_map
