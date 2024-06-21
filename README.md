
# neuralib

[![Document Status](https://readthedocs.org/projects/neuralib/badge/?version=latest)](https://neuralib.readthedocs.io/en/latest/index.html)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/neura-library)
[![PyPI version](https://badge.fury.io/py/neura-library.svg)](https://badge.fury.io/py/neura-library)
[![Downloads](https://static.pepy.tech/badge/neura-library)](https://pepy.tech/project/neura-library)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/ytsimon2004/neuralib)

## Utility tools for rodent system neuroscience research, including Open Source Wrapper or Parser

## See the [Documentation ](https://neuralib.readthedocs.io/en/latest/index.html)

# Installation

- `pip install neura-library` in your conda env with Python ~= 3.9.0
- Checkout [Release notes](https://github.com/ytsimon2004/neuralib/releases)
- According to purpose, install the optional package [requirements-optional.txt](requirements-optional.txt)

----------------------------

# Open-Source tools API call / data parsing

## atlas

- Module for whole brain, slice view visualization and rois classification
  - `neuralib.atlas.ccf`: Customized hierarchical classification for the mouse brain atlas
  - `neuralib.atlas.brainrender`: cli-based wrapper for [brainrender](https://github.com/brainglobe/brainrender)
  - `neuralib.atlas.cellatlas`: Volume and cell types counts for each brain region, refer
    to [Cell Atlas](https://portal.bluebrain.epfl.ch/resources/models/cell-atlas/)
  - `neuralib.atlas.ibl`: Slice view plotting wrapper
    for [ibllib](https://github.com/int-brain-lab/ibllib?tab=readme-ov-file)
    and [iblatlas](https://int-brain-lab.github.io/iblenv/_autosummary/ibllib.atlas.html)

## calimg

- Module for 2photon calcium imaging acquisition and result parsing
  - `neuralib.calimg.scan_image`: Data acquired from [ScanImage](https://www.mbfbioscience.com/products/scanimage/) (
    under
    DEV)
  - `neuralib.calimg.scanbox`: Data acquired from [Scanbox](https://scanbox.org/tag/two-photon/)
  - `neuralib.calimg.suite2p`:  Result parser for [suite2p](https://github.com/MouseLand/suite2p)

## segmentation

- Module for neuronal segmentation
  - `neuralib.segmentation.cellpose`: Result Parser and batch running
    for [cellpose](https://github.com/MouseLand/cellpose)
  - `neuralib.segmentation.stardist`: Result Parser and batch running
    for [stardist](https://github.com/stardist/stardist)

## wrapper

- Module for other open-source tools wrapper
  - `neuralib.wrapper.deeplabcut`: Result parser for [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)
  - `neuralib.wrapper.facemap`: Result parser for [facemap](https://github.com/MouseLand/facemap)
  - `neuralib.wrapper.rastermap`: Run and result parser for [rastermap](https://github.com/MouseLand/rastermap)

----------------------------

# Utilities Modules

## argp

- `neuralib.argp`: Use argparse as dataclass field

## persistence

- `neuralib.persistence`: caching the analyzed results (i.e., concatenation for statistic purpose)

## bokeh_model

- `neuralib.bokeh_model`: Interactive dashboard visualization

## sqlp

- `neuralib.sqlp`: Python functions to build a SQL (sqlite3) statement.

--------------------------

# Others

- `neuralib.plot`: Module for general plotting purpose


- `neuralib.model.bayes_decoding`: Position decoding using population neuronal activity


- `neuralib.tools.imglib`: Image processing library (under DEV)
- `neuralib.tools.slack_bot`: Real-time slack notification bot for analysis pipeline


- `neuralib.util.cli_args`: run script as subprocess
- `neuralib.util.color_logging`: logging with color format
- `neuralib.util.csv`: csv context manager
- `neuralib.util.gpu`: OS-dependent gpu info
- `neuralib.util.profile_test`: simple benchmark profile testing
- `neuralib.util.table`: rich table visualization

## project.scripts using CLI

### `brender`

- see examples in [api](https://neuralib.readthedocs.io/en/latest/api/neuralib.atlas.brainrender.html)



