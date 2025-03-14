# neuralib

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/neura-library)
[![PyPI version](https://badge.fury.io/py/neura-library.svg)](https://badge.fury.io/py/neura-library)
[![Downloads](https://static.pepy.tech/badge/neura-library)](https://pepy.tech/project/neura-library)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Document Status](https://readthedocs.org/projects/neuralib/badge/?version=latest)](https://neuralib.readthedocs.io/en/latest/index.html)

## Utility tools for rodent system neuroscience research, including Open Source Wrapper or Parser

## See the [Documentation ](https://neuralib.readthedocs.io/en/latest/index.html) and [Examples](https://github.com/ytsimon2004/neuralib/tree/main/doc/source/notebooks)

## Checkout [Release notes](https://github.com/ytsimon2004/neuralib/releases)

# Installation

## conda environment

- Create and activate a new conda environment (Python >=3.10, but >=3.12 not yet tested), then install:

```shell
conda create -n neuralib python=3.10
conda activate neuralib
pip install neura-library
```

- If you wish to install **all dependencies**, run:

```shell
pip install neura-library[all]
```

- If you wish to install the **minimal required dependencies** according to usage purpose:

Choices in []: atlas, scanner, imaging, segmentation, model, track, gpu, profile, imagelib, tools, full
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

- Make sure python version (>=3.10, but >=3.12 not yet tested), both in `pyproject.py` and `.python-version`

```shell
uv python install 3.10
```

- If you wish to install **all dependencies**, run:

```shell
uv add neura-library[all]
```

- If you wish to install the **minimal required dependencies** according to usage purpose:

Choices in []: atlas, scanner, imaging, segmentation, model, track, gpu, profile, imagelib, tools, full
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
