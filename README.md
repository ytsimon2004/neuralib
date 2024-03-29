# neuralib
- Utility tools for system neuroscience research, including Open Source Wrapper or Parser

## Installation

- Before pip release, clone the repo
- Run `pip install .` in the base directory, if develop mode, run `pip install -e .`

## Usage
### argp
- `neuralib.argp`: Use argparse as dataclass field

### atlas
- Module for whole brain & Slice view visualization
- `neuralib.atlas.brainrender`: cli-based wrapper for [brainrender](https://github.com/brainglobe/brainrender)
- `neuralib.atlas.cellatlas`: Volume and cell types counts for each brain region, refer to [Cell Atlas](https://portal.bluebrain.epfl.ch/resources/models/cell-atlas/)
- `neuralib.atlas.ibl`: Slice view plotting wrapper for [ibllib](https://github.com/int-brain-lab/ibllib?tab=readme-ov-file)

### bokeh_model
- `neuralib.bokeh_model`: Interactive dashboard visualization

### calimg
- Module for 2photon calcium imaging acquisition and result parsing
- `neuralib.calimg.scan_image`: Data acquired from [ScanImage](https://www.mbfbioscience.com/products/scanimage/) (under DEV)
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
- `neuralib.segmentation.cellpose`: Result Parser and batch running for [cellpose](https://github.com/MouseLand/cellpose)
- `neuralib.segmentation.stardist`: Result Parser and batch running for [stardist](https://github.com/stardist/stardist)

### stimpy
- Tools and Result parser for visual-guided behavior dataset
- Acquisition system are currently only lab internal usage

### tools
- `neuralib.tools.imglib`: Image processing library (under DEV)
- `neuralib.tools.slack_bot`: Real-time slack notification bot for analysis pipeline


### wrapper
- Module for other open-source tools wrapper
- `neuralib.tools.facemap`: Result parser for [facemap](https://github.com/MouseLand/facemap)
- `neuralib.tools.facemap`: Run and result parser for [rastermap](https://github.com/MouseLand/rastermap)


### util
- TODO