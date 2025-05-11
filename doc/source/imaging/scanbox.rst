ScanBox
========================

This module simply provides

1. result parser for scanbox acquisition system

2. view the sequence / save as tiff file

- **Refer to API**: :mod:`neuralib.imaging.scanbox`


Metadata information
----------------------

.. code-block:: python

    from neuralib.imaging.scanbox import read_scanbox

    file = ...  # scanbox .mat output file
    sbx = read_scanbox(file)
    print(sbx.asdict())  # print the information as dictionary



Metadata to Json
-------------------

.. code-block:: python

   from neuralib.imaging.scanbox import sbx_to_json

   file = ...  # scanbox .mat output file
   output_file = ...  # *.json
   sbx_to_json(file, outputfile)




Screen Shot file to tiff
-------------------------

.. code-block:: python

    from neuralib.imaging.scanbox import screenshot_to_tiff

    file = ...  # scanbox .mat screenshot output file
    output = ... # *.tiff
    screenshot_to_tiff(file, output)



ScanBox Imaging View
-------------------------

Directly view the image sequence as mmap


Use CLI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ python -m neuralib.imaging.scanbox.view -h


.. code-block:: text

    usage: view.py [-h] [--frames FRAMES] [--plane PLANE] [--channel CHANNEL] [--verbose] [--show] [--tiff TO_TIFF] PATH

    positional arguments:
      PATH               directory containing .sbx/.mat scanbox output

    options:
      -h, --help         show this help message and exit
      --frames FRAMES    indices of image sequences, if None, then all frames
      --plane PLANE      which optic plane
      --channel CHANNEL  which pmt channel
      --verbose          show meta verbose
      --show             play the selected imaging sequences
      --tiff TO_TIFF     save sequence as tiff output


Example playing the 100-200 frames

.. code-block:: bash

    $ python -m neuralib.imaging.scanbox.view PATH -P <OPTIC_PLANE> -C <PMT_CHANNEL> -F 100,200


Example save 100-200 Frames as tiff

.. code-block:: bash

    $ python -m neuralib.imaging.scanbox.view PATH -P <OPTIC_PLANE> -C <PMT_CHANNEL> -F 100,200 -O test.tiff



Use API call
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from neuralib.imaging.scanbox.view import ScanBoxView

    directory  = ...  # directory contain the .sbx and .mat output from scanbox
    view = ScanBoxView(directory)

    # play
    view.show(slice(100,200), plane=0, channel=0)

    # save as tiff
    view.to_tiff(slice(100,200), plane=0, channel=0, output='test.tiff')

