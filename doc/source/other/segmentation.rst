Cellular Segmentation
=======================


Provide CLI for cellular segmentation and visualization

Design mainly for quick visualization of results and batch processing (i.e., whole-brain image), and save as imageJ/Fiji ``.roi``.


StarDist
-------------

- **Refer to API**: :mod:`neuralib.segmentation.stardist`


**Example of run an image segmentation in 2D mode and visualize using napari**

.. code-block:: bash

    $ python -m neuralib.segmentation.stardist.run_2d -F <IMAGE_FILE> --napari

- It produces an output ``<IMAGE_FILE>.npz`` in the same directory, If the output exists, the napari will directly use the output. If need run again and rewrite the output, use ``--force-eval`` option



**Example of run batch image segmentation in 2D mode from a directory (.tif files)**

.. code-block:: bash

    $ python -m neuralib.segmentation.stardist.run_2d -D <DIRECTORY> --suffix .tif

- It produces multiple ``*.npz`` in the same directory.


**See help using** ``-h`` **option**

.. code-block:: bash

    $ python -m neuralib.segmentation.stardist.run_2d -h


**Output**

.. code-block:: text

    usage: run_2d.py [-h] [--model 2D_versatile_fluo|2D_versatile_he|2D_paper_dsb2018|2D_demo] [--invalid] [--no_norm] [--napari]
                     [--prob PROB_THRESH] [--file FILE | --dir DIRECTORY] [--dir_suffix .tif|.tiff|.png] [--save_roi]

    Run the Stardist model for segmentation

    options:
      -h, --help            show this help message and exit
      --model 2D_versatile_fluo|2D_versatile_he|2D_paper_dsb2018|2D_demo
                            stardist pretrained model
      --invalid             force re-evaluate the result
      --no_norm             not do percentile-based image normalization
      --napari              view result by napari GUI, only available in single file mode
      --prob PROB_THRESH    Consider only object candidates from pixels with predicted object probability above this threshold. Seealso: stardist.models.base._predict_instances_generator: prob_thresh

    Data I/O Options:
      --file FILE, --image_path FILE
                            image file path
      --dir DIRECTORY       directory for batch imaging processing
      --dir_suffix .tif|.tiff|.png
                            suffix in the directory for batch mode
      --save_rois           if save also the imageJ/Fiji compatible .roi file




Cellpose
------------------

- **Refer to API**: :mod:`neuralib.segmentation.cellpose`

The recent version of `Cellpose <https://github.com/MouseLand/cellpose>`_ (``v4.0.1``) provided better CLI,

Thus deprecate the wrapper usage from neuralib, only provide basic output parser and helper functions


**Once install the native cellpose package, run**

.. code-block:: bash

    $ python -m cellpose -h


**Output**

.. code-block:: text

    usage: cellpose [-h] [--version] [--verbose] [--Zstack] [--use_gpu] [--gpu_device GPU_DEVICE] [--dir DIR]
                    [--image_path IMAGE_PATH] [--look_one_level_down] [--img_filter IMG_FILTER]
                    [--channel_axis CHANNEL_AXIS] [--z_axis Z_AXIS] [--chan CHAN] [--chan2 CHAN2] [--invert]
                    [--all_channels] [--pretrained_model PRETRAINED_MODEL] [--add_model ADD_MODEL]
                    [--pretrained_model_ortho PRETRAINED_MODEL_ORTHO] [--restore_type RESTORE_TYPE] [--chan2_restore]
                    [--transformer] [--no_norm] [--norm_percentile VALUE1 VALUE2] [--do_3D] [--diameter DIAMETER]
                    [--stitch_threshold STITCH_THRESHOLD] [--min_size MIN_SIZE] [--flow3D_smooth FLOW3D_SMOOTH]
                    [--flow_threshold FLOW_THRESHOLD] [--cellprob_threshold CELLPROB_THRESHOLD] [--niter NITER]
                    [--anisotropy ANISOTROPY] [--exclude_on_edges] [--augment] [--batch_size BATCH_SIZE] [--no_resample]
                    [--no_interp] [--save_png] [--save_tif] [--output_name OUTPUT_NAME] [--no_npy] [--savedir SAVEDIR]
                    [--dir_above] [--in_folders] [--save_flows] [--save_outlines] [--save_rois] [--save_txt] [--save_mpl]
                    [--train] [--test_dir TEST_DIR] [--file_list FILE_LIST] [--mask_filter MASK_FILTER]
                    [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] [--n_epochs N_EPOCHS]
                    [--train_batch_size TRAIN_BATCH_SIZE] [--bsize BSIZE] [--nimg_per_epoch NIMG_PER_EPOCH]
                    [--nimg_test_per_epoch NIMG_TEST_PER_EPOCH] [--min_train_masks MIN_TRAIN_MASKS] [--SGD SGD]
                    [--save_every SAVE_EVERY] [--model_name_out MODEL_NAME_OUT] [--diam_mean DIAM_MEAN] [--train_size]

    Cellpose Command Line Parameters

    options:
      -h, --help            show this help message and exit
      --version             show cellpose version info
      --verbose             show information about running and settings and save to log
      --Zstack              run GUI in 3D mode

    Hardware Arguments:
      --use_gpu             use gpu if torch with cuda installed
      --gpu_device GPU_DEVICE
                            which gpu device to use, use an integer for torch, or mps for M1

    Input Image Arguments:
      --dir DIR             folder containing data to run or train on.
      --image_path IMAGE_PATH
                            if given and --dir not given, run on single image instead of folder (cannot train with this
                            option)
      --look_one_level_down
                            run processing on all subdirectories of current folder
      --img_filter IMG_FILTER
                            end string for images to run on
      --channel_axis CHANNEL_AXIS
                            axis of image which corresponds to image channels
      --z_axis Z_AXIS       axis of image which corresponds to Z dimension
      --chan CHAN           Deprecated in v4.0.1+, not used.
      --chan2 CHAN2         Deprecated in v4.0.1+, not used.
      --invert              Deprecated in v4.0.1+, not used.
      --all_channels        Deprecated in v4.0.1+, not used.

    Model Arguments:
      --pretrained_model PRETRAINED_MODEL
                            model to use for running or starting training
      --add_model ADD_MODEL
                            model path to copy model to hidden .cellpose folder for using in GUI/CLI
      --pretrained_model_ortho PRETRAINED_MODEL_ORTHO
                            Deprecated in v4.0.1+, not used.
      --restore_type RESTORE_TYPE
                            Deprecated in v4.0.1+, not used.
      --chan2_restore       Deprecated in v4.0.1+, not used.
      --transformer         use transformer backbone (pretrained_model from Cellpose3 is transformer_cp3)

    Algorithm Arguments:
      --no_norm             do not normalize images (normalize=False)
      --norm_percentile VALUE1 VALUE2
                            Provide two float values to set norm_percentile (e.g., --norm_percentile 1 99)
      --do_3D               process images as 3D stacks of images (nplanes x nchan x Ly x Lx
      --diameter DIAMETER   use to resize cells to the training diameter (30 pixels)
      --stitch_threshold STITCH_THRESHOLD
                            compute masks in 2D then stitch together masks with IoU>0.9 across planes
      --min_size MIN_SIZE   minimum number of pixels per mask, can turn off with -1
      --flow3D_smooth FLOW3D_SMOOTH
                            stddev of gaussian for smoothing of dP for dynamics in 3D, default of 0 means no smoothing
      --flow_threshold FLOW_THRESHOLD
                            flow error threshold, 0 turns off this optional QC step. Default: 0.4
      --cellprob_threshold CELLPROB_THRESHOLD
                            cellprob threshold, default is 0, decrease to find more and larger masks
      --niter NITER         niter, number of iterations for dynamics for mask creation, default of 0 means it is
                            proportional to diameter, set to a larger number like 2000 for very long ROIs
      --anisotropy ANISOTROPY
                            anisotropy of volume in 3D
      --exclude_on_edges    discard masks which touch edges of image
      --augment             tiles image with overlapping tiles and flips overlapped regions to augment
      --batch_size BATCH_SIZE
                            inference batch size. Default: 8
      --no_resample         Deprecated in v4.0.1+, not used.
      --no_interp           do not interpolate when running dynamics (was default)

    Output Arguments:
      --save_png            save masks as png
      --save_tif            save masks as tif
      --output_name OUTPUT_NAME
                            suffix for saved masks, default is _cp_masks, can be empty if `savedir` used and different of
                            `dir`
      --no_npy              suppress saving of npy
      --savedir SAVEDIR     folder to which segmentation results will be saved (defaults to input image directory)
      --dir_above           save output folders adjacent to image folder instead of inside it (off by default)
      --in_folders          flag to save output in folders (off by default)
      --save_flows          whether or not to save RGB images of flows when masks are saved (disabled by default)
      --save_outlines       whether or not to save RGB outline images when masks are saved (disabled by default)
      --save_rois           whether or not to save ImageJ compatible ROI archive (disabled by default)
      --save_txt            flag to enable txt outlines for ImageJ (disabled by default)
      --save_mpl            save a figure of image/mask/flows using matplotlib (disabled by default). This is slow,
                            especially with large images.

    Training Arguments:
      --train               train network using images in dir
      --test_dir TEST_DIR   folder containing test data (optional)
      --file_list FILE_LIST
                            path to list of files for training and testing and probabilities for each image (optional)
      --mask_filter MASK_FILTER
                            end string for masks to run on. use '_seg.npy' for manual annotations from the GUI. Default:
                            _masks
      --learning_rate LEARNING_RATE
                            learning rate. Default: 1e-05
      --weight_decay WEIGHT_DECAY
                            weight decay. Default: 0.1
      --n_epochs N_EPOCHS   number of epochs. Default: 100
      --train_batch_size TRAIN_BATCH_SIZE
                            training batch size. Default: 1
      --bsize BSIZE         block size for tiles. Default: 256
      --nimg_per_epoch NIMG_PER_EPOCH
                            number of train images per epoch. Default is to use all train images.
      --nimg_test_per_epoch NIMG_TEST_PER_EPOCH
                            number of test images per epoch. Default is to use all test images.
      --min_train_masks MIN_TRAIN_MASKS
                            minimum number of masks a training image must have to be used. Default: 5
      --SGD SGD             Deprecated in v4.0.1+, not used - AdamW used instead.
      --save_every SAVE_EVERY
                            number of epochs to skip between saves. Default: 100
      --model_name_out MODEL_NAME_OUT
                            Name of model to save as, defaults to name describing model architecture. Model is saved in the
                            folder specified by --dir in models subfolder.
      --diam_mean DIAM_MEAN
                            Deprecated in v4.0.1+, not used.
      --train_size          Deprecated in v4.0.1+, not used.

