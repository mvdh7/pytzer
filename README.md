# Pytzer

![Tests](https://github.com/mvdh7/pytzer/workflows/Tests/badge.svg)
[![Coverage](https://github.com/mvdh7/pytzer/blob/jax/.misc/coverage.svg)](https://github.com/mvdh7/pytzer/blob/jax/.misc/coverage.txt)
[![pypi badge](https://img.shields.io/pypi/v/pytzer.svg?style=popout)](https://pypi.org/project/pytzer/)
[![Docs](https://readthedocs.org/projects/pytzer/badge/?version=jax&style=flat)](https://pytzer.readthedocs.io/en/jax/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Pytzer is a Python implementation of the Pitzer model for chemical activities in aqueous solutions [[P91](https://pytzer.readthedocs.io/en/jax/refs/#p)] plus solvers to determine the equilibrium state of the system.

## Installation

Due to its dependency on [JAX](https://github.com/google/jax), Pytzer can only be installed on Unix systems, although it does work on Windows via [WSL](https://docs.microsoft.com/en-us/windows/wsl/).

### For development

Use the [environment.yml](https://github.com/mvdh7/pytzer/blob/jax/environment.yml) file to create a new environment with Conda:

    conda env create -f environment.yml

If you want to run this environment in Spyder v5, you will also need to upgrade `spyder-kernels` (the default version is suitable for Spyder v4):

    conda update -n pytzer -c conda-forge spyder-kernels

Finally, fork and/or clone this repo to somewhere that your Python can see it.

### For general use

At present only up to v0.4.3 is available for installation with `pip` using:

    pip install pytzer

Once v0.5+ is available via `pip`, you will also need to set the environment variable `JAX_ENABLE_X64=True`, unless you have built the environment using the environment.yml file from the section above.

## Documentation

A work in progress at [pytzer.readthedocs.io/en/jax/](https://pytzer.readthedocs.io/en/jax/).

Pytzer is maintained by [Dr Matthew P. Humphreys](https://mvdh.xyz) at the [NIOZ Royal Netherlands Institute for Sea Research](https://www.nioz.nl/en) (Texel, the Netherlands).
