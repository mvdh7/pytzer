# Pytzer

![Tests](https://github.com/mvdh7/pytzer/workflows/Tests/badge.svg)
[![Coverage](https://github.com/mvdh7/pytzer/blob/main/.misc/coverage.svg)](https://github.com/mvdh7/pytzer/blob/main/.misc/coverage.txt)
[![pypi badge](https://img.shields.io/pypi/v/pytzer.svg?style=popout)](https://pypi.org/project/pytzer/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.2637914-informational)](https://doi.org/10.5281/zenodo.2637914)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Pytzer is a Python implementation of the Pitzer model for chemical activities in aqueous solutions [[P91](https://mvdh.xyz/pytzer/refs/#p)] plus solvers to determine the equilibrium state of the system.

**Pytzer is in beta!  Use at your own peril.**

- [Pytzer](#pytzer)
  - [Installation](#installation)
    - [For general use](#for-general-use)
    - [For development](#for-development)
  - [Documentation](#documentation)
  - [Citation](#citation)

## Installation

Due to its dependency on [JAX](https://github.com/google/jax), Pytzer can only be installed on Unix systems, although it does work on Windows via [WSL](https://docs.microsoft.com/en-us/windows/wsl/).

### For general use

Install with pip:

    pip install pytzer

Note that you should also install the dependencies (especially NumPy) using pip, not conda, for best performance.  This happens automatically with the above command if the dependencies are not already installed.

Once installed, you will need to set the environment variable `JAX_ENABLE_X64=True`.  For example, using conda:

    conda env config vars set JAX_ENABLE_X64=True

## Documentation

A work in progress at [mvdh.xyz/pytzer](https://mvdh.xyz/pytzer).

## Citation

Pytzer is maintained by [Dr Matthew P. Humphreys](https://seaco2.group) at the [NIOZ Royal Netherlands Institute for Sea Research](https://www.nioz.nl/en) (Texel, the Netherlands).

For now, the appropriate citation is:

> Humphreys, Matthew P. and Schiller, Abigail J. (2023). Pytzer: the Pitzer model for chemical activities and equilibria in aqueous solutions in Python (beta).  *Zenodo.*  [doi:10.5281/zenodo.2637914](https://doi.org/10.5281/zenodo.2637914).

Please report which version of Pytzer you are using.  You can find this in Python with:

```python
import pytzer as pz
pz.hello()
```
