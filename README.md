
# neuralib

[![Document Status](https://readthedocs.org/projects/neuralib/badge/?version=latest)](https://neuralib.readthedocs.io/en/latest/index.html)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/neura-library)
[![PyPI version](https://badge.fury.io/py/neura-library.svg)](https://badge.fury.io/py/neura-library)
[![Downloads](https://static.pepy.tech/badge/neura-library)](https://pepy.tech/project/neura-library)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/ytsimon2004/neuralib)

- Utility tools for rodent system neuroscience research, including Open Source Wrapper or Parser

## Installation

- `pip install neura-library` in your conda env with Python >= 3.9
- According to purpose, install the optional package [requirements-optional.txt](requirements-optional.txt)

## Usage

## See to [Doc](https://neuralib.readthedocs.io/en/latest/index.html)

### argp

- `neuralib.argp`: Use argparse as dataclass field

### atlas

- Module for whole brain, slice view visualization and rois classification
- `neuralib.atlas.ccf`: Customized hierarchical classification for the mouse brain atlas
- `neuralib.atlas.brainrender`: cli-based wrapper for [brainrender](https://github.com/brainglobe/brainrender)
- `neuralib.atlas.cellatlas`: Volume and cell types counts for each brain region, refer
  to [Cell Atlas](https://portal.bluebrain.epfl.ch/resources/models/cell-atlas/)
- `neuralib.atlas.ibl`: Slice view plotting wrapper
  for [ibllib](https://github.com/int-brain-lab/ibllib?tab=readme-ov-file)
  and [iblatlas](https://int-brain-lab.github.io/iblenv/_autosummary/ibllib.atlas.html)

### bokeh_model

- `neuralib.bokeh_model`: Interactive dashboard visualization

### calimg

- Module for 2photon calcium imaging acquisition and result parsing
- `neuralib.calimg.scan_image`: Data acquired from [ScanImage](https://www.mbfbioscience.com/products/scanimage/) (under
  DEV)
- `neuralib.calimg.scanbox`: Data acquired from [Scanbox](https://scanbox.org/tag/two-photon/)
- `neuralib.calimg.suite2p`:  Result parser for [suite2p](https://github.com/MouseLand/suite2p)

### model

- under DEV

### persistence

- Module for caching the analyzed results (i.e., concatenation for statistic purpose)

### plot

- Module for general plotting purpose

### scanner

- Module for parsing Zeiss confocal scanning data
- `neuralib.scanner.czi`: .czi data format
- `neuralib.scanner.lsm`: .lsm data format

### segmentation

- Module for cellular segmentation
- `neuralib.segmentation.cellpose`: Result Parser and batch running
  for [cellpose](https://github.com/MouseLand/cellpose)
- `neuralib.segmentation.stardist`: Result Parser and batch running for [stardist](https://github.com/stardist/stardist)

### stimpy

- Tools and Result parser for visual-guided behavior dataset
- Acquisition system are currently only lab internal usage

### tools

- `neuralib.tools.imglib`: Image processing library (under DEV)
- `neuralib.tools.slack_bot`: Real-time slack notification bot for analysis pipeline

### wrapper

- Module for other open-source tools wrapper
- `neuralib.tools.deeplabcut`: Result parser for [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)
- `neuralib.tools.facemap`: Result parser for [facemap](https://github.com/MouseLand/facemap)
- `neuralib.tools.rastermap`: Run and result parser for [rastermap](https://github.com/MouseLand/rastermap)

### util

- `cli_args`: run script as subprocess
- `color_logging`: logging with color format
- `csv`: csv context manager
- `gpu`: OS-dependent gpu info
- `profile_test`: simple benchmark profile testing
- `table`: rich table visualization

## project.scripts using cli

### `brender`

- see examples in [api](https://neuralib.readthedocs.io/en/latest/api/neuralib.atlas.brainrender.html)


