# neuralib

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/neura-library)
[![PyPI version](https://badge.fury.io/py/neura-library.svg)](https://badge.fury.io/py/neura-library)
[![Downloads](https://static.pepy.tech/badge/neura-library)](https://pepy.tech/project/neura-library)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Document Status](https://readthedocs.org/projects/neuralib/badge/?version=latest)](https://neuralib.readthedocs.io/en/latest/index.html)

## Utility tools for rodent system neuroscience research, including Open Source Wrapper or Parser

## See the [Documentation ](https://neuralib.readthedocs.io/en/latest/index.html) and [Examples](doc/source/notebooks)

## Checkout [Release notes](https://github.com/ytsimon2004/neuralib/releases)

# Installation

## conda environment

- Create and activate a new conda environment (Python >=3.9, but >=3.11 not yet tested), then install:

```shell
conda create -n neuralib python=3.9
conda activate neuralib
pip install neura-library
```

- If you wish to install **all dependencies**, run:

```shell
pip install neura-library[all]
```

- If you wish to install the **minimal required dependencies** according to usage purpose:

Choices in []: atlas, scanner, calimg, segmentation, model, track, gpu, profile, imagelib, tools, full
Example of using the atlas module:

```shell
pip install neura-library[atlas]
```

- If installing in developer mode (Install pre-commit and linter check by ruff)

```shell
pip install neura-library[dev]
pre-commit install
ruff check .
```

## uv virtual environment

- Install uv, run in Unix or git bash (Windows):

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- Follow uv project structure [doc](https://docs.astral.sh/uv/guides/projects/#creating-a-new-project):

```shell
uv init
```

- Make sure python version (>=3.9, but >=3.11 not yet tested), both in `pyproject.py` and `.python-version`

```shell
uv python install 3.9
```

- If you wish to install **all dependencies**, run:

```shell
uv add neura-library[all]
```

- If you wish to install the **minimal required dependencies** according to usage purpose:

Choices in []: atlas, scanner, calimg, segmentation, model, track, gpu, profile, imagelib, tools, full
Example of using the atlas module:

```shell
uv add neura-library[atlas]
```

- If installing in developer mode (Install pre-commit and linter check by ruff)

```shell
uv add neura-library[dev]
pre-commit install
ruff check .
```

----------------------------

# Module

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
  - `neuralib.calimg.spikes`: dF/F to spike activity (OASIS/Cascade)

## segmentation

- Module for neuronal segmentation
    - `neuralib.segmentation.cellpose`: Result Parser and batch running
      for [cellpose](https://github.com/MouseLand/cellpose)
    - `neuralib.segmentation.stardist`: Result Parser and batch running
      for [stardist](https://github.com/stardist/stardist)

## model

- `neuralib.model.bayes_decoding`: Position decoding using population neuronal activity

- `neuralib.model.rastermap`: Run and result parser for [rastermap](https://github.com/MouseLand/rastermap)

## morpho

- Morphological reconstruction data presentation

## tracking

- Module for other open-source tools wrapper
    - `neuralib.tracking.deeplabcut`: Result parser for [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)

    - `neuralib.tracking.facemap`: Result parser for [facemap](https://github.com/MouseLand/facemap)

## Utilities Modules

- `neuralib.argp`: Use argparse as dataclass field

- `neuralib.persistence`: caching the analyzed results (i.e., concatenation for statistic purpose)

- `neuralib.bokeh_model`: Interactive dashboard visualization

- `neuralib.sqlp`: Python functions to build a SQL (sqlite3) statement.

- `neuralib.plot`: Module for general plotting purpose

- `neuralib.imglib`: Image processing library (under DEV)

- `neuralib.tools.gspread`: Google spreadsheet API wrapper for read/write

- `neuralib.tools.slack_bot`: Real-time slack notification bot for analysis pipeline

- `neuralib.util.cli_args`: run script as subprocess

- `neuralib.util.color_logging`: logging with color format

- `neuralib.util.gpu`: OS-dependent gpu info

- `neuralib.util.profile_test`: simple benchmark profile testing

- `neuralib.util.table`: rich table visualization

- `neuralib.util.segments`: Epoch or value segments

# CLI project.scripts

- `brender`: see examples in [api](https://neuralib.readthedocs.io/en/latest/api/neuralib.atlas.brainrender.html)
