# Pytzer

![Tests](https://github.com/mvdh7/pytzer/workflows/Tests/badge.svg)
[![Coverage](img/coverage.svg)](https://github.com/mvdh7/pytzer/blob/master/.misc/coverage.txt)
[![pypi badge](https://img.shields.io/pypi/v/pytzer.svg?style=popout)](https://pypi.org/project/pytzer/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.2637914-informational)](https://doi.org/10.5281/zenodo.2637914)
[![Docs](https://readthedocs.org/projects/pytzer/badge/?version=latest&style=flat)](https://pytzer.readthedocs.io/en/latest/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Pytzer is a Python implementation of the Pitzer model for chemical activities in aqueous solutions [[P91](refs/#p)] plus solvers to determine the equilibrium state of the system.

## Installation

Consult the [README.md on GitHub](https://github.com/mvdh7/pytzer/tree/master#pytzer) for up-to-date installation instructions.

## Development status

Pytzer is in beta.  Tests of the accuracy of its parameters and equations are underway, so results may change.  Its API may change, and modules and functions may be added, removed and renamed.  Use at your own peril!

## Citation

A manuscript describing Pytzer is in preparation.  In the meantime, please cite:

> Humphreys, Matthew P. and Schiller, Abigail J. (2021).  Pytzer: the Pitzer model for chemical activities and equilibria in aqueous solutions in Python (beta).  *Zenodo.* [doi:10.5281/zenodo.2637914](https://doi.org/10.5281/zenodo.2637914).

## Acknowledgements

Pytzer is maintained by [Dr Matthew P. Humphreys](https://humphreys.science) at the NIOZ Royal Netherlands Institute for Sea Research (Texel, the Netherlands).  Its initial development at the University of East Anglia was funded indirectly by the Natural Environment Research Council (NERC, UK).

Pytzer contains many functions and coefficients representing the effects of different solute interactions on solution properties that have been empirically determined from painstaking experiments and data compilations by hundreds of researchers over the course at least a century.  We have done our best to list the small selection of this enormous body of work brought together here in the [references](refs).

## License

Pytzer is licensed under the [GNU General Public License version 3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html).
