from io import BytesIO

import numpy as np
import requests
from PIL import Image

from neuralib.imglib.factory import ImageProcFactory
from neuralib.plot import plot_figure


def get_image_url(*urls) -> np.ndarray:
    arr = []
    for url in urls:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        arr.append(np.array(img))

    return np.hstack(arr)


def test_image_factory():
    atlas_example = 'https://connectivity.brain-map.org/tiles//external/connectivity/prod16/0500156609/0500156609.aff/TileGroup5/'
    lh = atlas_example + '5-1-0.jpg?siTop=13568&siLeft=160&siWidth=16368&siHeight=11056&filter=rgb&filterVals=0.5,0.5,0.5'
    rh = atlas_example + '5-2-0.jpg?siTop=13568&siLeft=160&siWidth=16368&siHeight=11056&filter=rgb&filterVals=0.5,0.5,0.5'

    image = get_image_url(lh, rh)
    ipf = ImageProcFactory(image)

    with plot_figure(None, 3, 3, figsize=(10, 10), sharex=True, sharey=True) as _ax:
        #
        ax = _ax[0, 0]
        ax.imshow(ipf.image)
        ax.set_title('raw')

        #
        ax = _ax[0, 1]
        ax.imshow(ipf.view_2d(flip=False).image)
        ax.set_title('view 2d')

        #
        ax = _ax[0, 2]
        ax.imshow(ipf.cvt_gray().image, cmap='binary')
        ax.set_title('grayscale')

        #
        ax = _ax[1, 0]
        ax.imshow(ipf.select_channel('red').image, cmap='Reds')
        ax.set_title('red channel')

        #
        ax = _ax[1, 1]
        ax.imshow(ipf.select_channel('b').image, cmap='Blues')
        ax.set_title('blue channel')

        #
        ax = _ax[1, 2]
        ax.imshow(ipf
                  .select_channel('b')
                  .edge_detection(lower_threshold=100, upper_threshold=120)
                  .image, cmap='binary')
        ax.set_title('blue edge detect')

        #
        ax = _ax[2, 0]
        ax.imshow(ipf.gaussian_blur(ksize=3, sigma=5).image)
        ax.set_title('blur detect')

        #
        ax = _ax[2, 1]
        ax.imshow(ipf.select_channel('b').de_noise().image, cmap='Blues')
        ax.set_title('blue de noise')

        #
        ax = _ax[2, 2]
        ax.imshow(ipf.select_channel('b').binarize(threshold=100).image, cmap='Blues')
        ax.set_title('blue binarize')


if __name__ == '__main__':
    test_image_factory()
