# Pytzer

![Tests](https://github.com/mvdh7/pytzer/workflows/Tests/badge.svg)
[![Coverage](https://github.com/mvdh7/pytzer/blob/jax/.misc/coverage.svg)](https://github.com/mvdh7/pytzer/blob/jax/.misc/coverage.txt)
[![pypi badge](https://img.shields.io/pypi/v/pytzer.svg?style=popout)](https://pypi.org/project/pytzer/)
[![Docs](https://readthedocs.org/projects/pytzer/badge/?version=jax&style=flat)](https://pytzer.readthedocs.io/en/jax/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Pitzer model for chemical activities and other properties of aqueous solutions.  Undergoing beta testing and development.

## Installation

Due to its dependency on JAX, Pytzer can only be installed on Unix systems, although it does work on Windows via [WSL](https://docs.microsoft.com/en-us/windows/wsl/).

### For development

Use the [environment.yml](https://github.com/mvdh7/pytzer/blob/jax/environment.yml) file to create a new environment with Conda:

    conda env create -f environment.yml

If you want to run this environment in Spyder v5, you will also need to upgrade `spyder-kernels`:

    conda update -n pytzer -c conda-forge spyder-kernels

### For general use

    pip install pytzer

You will also need to set the environment variable `JAX_ENABLE_X64=True`, unless you have built the environment using the environment.yml file from the section above.

## Documentation

A work in progress at [pytzer.readthedocs.io](https://pytzer.readthedocs.io/en/jax/).

Pytzer is maintained by [Dr Matthew P. Humphreys](https://mvdh.xyz) at the [NIOZ Royal Netherlands Institute for Sea Research](https://www.nioz.nl/en) (Texel, the Netherlands).
