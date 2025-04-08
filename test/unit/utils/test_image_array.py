import os
import tempfile
from io import BytesIO
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import requests
from PIL import Image

from neuralib.imglib.array import ImageArrayWrapper
from neuralib.imglib.array import image_array
from neuralib.plot import plot_figure


@pytest.fixture(scope="class")
def image() -> ImageArrayWrapper:
    atlas_example = 'https://connectivity.brain-map.org/tiles//external/connectivity/prod16/0500156609/0500156609.aff/TileGroup5/'
    lh = atlas_example + '5-1-0.jpg?siTop=13568&siLeft=160&siWidth=16368&siHeight=11056&filter=rgb&filterVals=0.5,0.5,0.5'
    rh = atlas_example + '5-2-0.jpg?siTop=13568&siLeft=160&siWidth=16368&siHeight=11056&filter=rgb&filterVals=0.5,0.5,0.5'
    image = get_image_url(lh, rh)
    return image_array(image)


def get_image_url(*urls) -> np.ndarray:
    arr = []
    for url in urls:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        arr.append(np.array(img))

    return np.hstack(arr)


class TestImageArrayWrapper:

    def test_path_io(self):
        dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
        dummy_bgr = cv2.cvtColor(dummy_img, cv2.COLOR_RGB2BGR)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name

        cv2.imwrite(temp_path, dummy_bgr)

        try:
            loaded = ImageArrayWrapper(temp_path)
            assert loaded.mode == 'RGB'
            assert loaded.shape == (10, 10, 3)
        finally:
            os.unlink(temp_path)

    def test_mode(self, image):
        assert image.mode == 'RGB'

    def test_to_gray(self, image):
        assert image.to_gray().ndim == 2

    def test_select_channel(self, image):
        assert image.select_channel('b').ndim == 2

    def test_view_2d(self, image):
        assert image.view_2d().ndim == 2

    # @pytest.mark.skip(reason='manually show')
    @patch("matplotlib.pyplot.show")
    def test_plot_basic(self, mock, image):
        with plot_figure(None, 3, 3, sharex=True, sharey=True) as ax:
            ax = ax.ravel()
            ax[0].imshow(image)
            ax[1].imshow(image.to_gray(), cmap='binary')
            ax[2].imshow(image.select_channel('r'), cmap='Reds')
            ax[3].imshow(image.select_channel('g'), cmap='Greens')
            ax[4].imshow(image.select_channel('b'), cmap='Blues')
            ax[5].imshow(image.view_2d())

    # @pytest.mark.skip(reason='manually show')
    @patch("matplotlib.pyplot.show")
    def test_plot_proc(self, mock, image: ImageArrayWrapper):
        with plot_figure(None, 3, 3, sharex=True, sharey=True) as ax:
            ax = ax.ravel()
            ax[0].imshow(image)
            ax[1].imshow(image.gaussian_blur(ksize=[3, 3], sigma_x=5, sigma_y=5), cmap='binary')
            ax[2].imshow(image.canny_filter())
            ax[3].imshow(image.binarize(thresh=15))
            ax[4].imshow(image.denoise())
            ax[4].imshow(image.enhance_contrast())
            ax[5].imshow(image.local_maxima('b'))
