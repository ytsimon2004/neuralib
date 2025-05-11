Image library
===============
This Module provide factory and utility for imaging processing

- **Refer to API**: :mod:`neuralib.imglib`

Array Factory
----------------------

This module defines the ImageArrayWrapper, a subclass of numpy.ndarray that wraps image data
and provides chainable image processing methods. It allows you to process images fluently
using a series of method calls, and then display or further analyze the results as a standard
NumPy array.

Example of Chainable Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from neuralib.imglib.processor import image_array
    import matplotlib.pyplot as plt

    # Load an image (from a file path or a NumPy array)
    img = image_array("path/to/image.jpg")

    # Process the image: convert to grayscale, apply Gaussian blur, then perform edge detection.
    processed = img.to_gray().gaussian_blur(ksize=[5, 5], sigma_x=2.0, sigma_y=2.0).canny_filter()

    # Display the processed image using matplotlib.
    plt.imshow(processed, cmap='gray')
    plt.title("Processed Image")
    plt.show()

Method Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - **Method**
     - **Description**
   * - :meth:`to_gray() <neuralib.imglib.array.ImageArrayWrapper.to_gray>`
     - Converts the image to grayscale.
   * - :meth:`flipud() <neuralib.imglib.array.ImageArrayWrapper.flipud>`
     - Flips the image vertically (upside down).
   * - :meth:`fliplr() <neuralib.imglib.array.ImageArrayWrapper.fliplr>`
     - Flips the image horizontally (left-to-right).
   * - :meth:`select_channel(channel) <neuralib.imglib.array.ImageArrayWrapper.select_channel>`
     - Extracts a specified color channel (e.g. 'r', 'g', or 'b') as a grayscale image.
   * - :meth:`view_2d(flipud=False) <neuralib.imglib.array.ImageArrayWrapper.view_2d>`
     - Converts a multi-channel image to a 2D representation (grayscale) with an option to flip vertically.
   * - :meth:`gaussian_blur(ksize, sigma_x, sigma_y) <neuralib.imglib.array.ImageArrayWrapper.gaussian_blur>`
     - Applies a Gaussian blur to the image using the specified kernel size and sigma values.
   * - :meth:`canny_filter(threshold_1, threshold_2) <neuralib.imglib.array.ImageArrayWrapper.canny_filter>`
     - Applies the Canny edge detection algorithm on the grayscale version of the image.
   * - :meth:`binarize(thresh, maxval) <neuralib.imglib.array.ImageArrayWrapper.binarize>`
     - Converts the image to a binary image using thresholding.
   * - :meth:`denoise(h, temp_win_size, search_win_size) <neuralib.imglib.array.ImageArrayWrapper.denoise>`
     - Denoises the image using non-local means denoising.
   * - :meth:`enhance_contrast() <neuralib.imglib.array.ImageArrayWrapper.enhance_contrast>`
     - Enhances the image contrast via histogram equalization.
   * - :meth:`local_maxima(channel) <neuralib.imglib.array.ImageArrayWrapper.local_maxima>`
     - Computes the local maxima on a specified color channel after channel extraction.



CV2 Labeller
----------------------


Simple CV2-based viewer/labeller GUI for image sequences

Use Cases:

- viewing the image sequences

- label each image and save as csv dataframe (human-eval for population neurons activity profile)


Load sequences from a directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **See Option**

.. code-block:: bash

    $ python -m neuralib.imglib.labeller -h

- **Example**

.. code-block:: bash

    $ python neuralib.imglib.labeller -D <DIR>


- **API call**

.. code-block:: python

    from neuralib.imglib.labeller import SequenceLabeller

    directory = ...
    labeller = SequenceLabeller.load_from_dir(directory)
    labeller.main()


Load sequences from sequences array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from neuralib.imglib.labeller import SequenceLabeller

    arr = ...   # numpy array with (F, H, W, <3>)
    labeller = SequenceLabeller.load_sequences(arr)
    labeller.main()