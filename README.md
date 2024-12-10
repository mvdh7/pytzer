# Pytzer

![Tests](https://github.com/mvdh7/pytzer/workflows/Tests/badge.svg)
[![Coverage](https://raw.githubusercontent.com/mvdh7/pytzer/develop/.misc/coverage.svg)](https://github.com/mvdh7/pytzer/blob/develop/.misc/coverage.txt)
[![pypi badge](https://img.shields.io/pypi/v/pytzer.svg?style=popout)](https://pypi.org/project/pytzer/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pytzer.svg?style-popout)](https://anaconda.org/conda-forge/pytzer)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.2637914-informational)](https://doi.org/10.5281/zenodo.2637914)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Pytzer is a Python implementation of the Pitzer model for chemical activities in aqueous solutions [[P91](https://pytzer.hseao3.group/refs/#p)] plus solvers to determine the equilibrium state of the system.

**Pytzer is in beta!  Use at your own peril.**

- [Pytzer](#pytzer)
  - [Installation](#installation)
    - [For general use](#for-general-use)
  - [Documentation](#documentation)
  - [Citation](#citation)

## Installation

### For general use

Install with pip (any OS):

    pip install pytzer

On Mac or Linux, you can alternatively install from conda-forge:

    conda install conda-forge::pytzer

However, the above will not work on Windows, because jax is not on conda-forge for Windows - so use pip instead.

Once installed, you will need to set the environment variable `JAX_ENABLE_X64=True`.  For example, using conda:

    conda env config vars set JAX_ENABLE_X64=True

## Documentation

A work in progress at [pytzer.hseao3.group](https://pytzer.hseao3.group).

## Citation

Pytzer is maintained by [Dr Matthew P. Humphreys](https://hseao3.group) at the [NIOZ Royal Netherlands Institute for Sea Research](https://www.nioz.nl/en) (Texel, the Netherlands).

For now, the appropriate citation is:

> Humphreys, Matthew P. and Schiller, Abigail J. (2024). Pytzer: the Pitzer model for chemical activities and equilibria in aqueous solutions in Python (beta).  *Zenodo.*  [doi:10.5281/zenodo.2637914](https://doi.org/10.5281/zenodo.2637914).

Please report which version of Pytzer you are using.  You can find this in Python with:

```python
import pytzer as pz
pz.hello()
```
