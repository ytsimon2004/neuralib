"""
Provide CLI for cellular segmentation and visualization, two packages are supported:

1. `Cellpose <https://github.com/MouseLand/cellpose>`_

- See usage :mod:`neuralib.segmentation.cellpose`

2. `Stardist <https://github.com/stardist/stardist>`_

- See usage in :mod:`neuralib.segmentation.stardist`


Design mainly for quick visualization of results and batch processing (i.e., whole-brain image).
If only small amount of images, **the native GUI are recommended**.


TODO
-----

- [ ] Fiji ``.roi`` dat converter

"""
