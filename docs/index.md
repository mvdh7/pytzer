# Pytzer

Pytzer is a Python implementation of the Pitzer model for chemical activities in aqueous solutions ([P91](refs/#p)) plus an equilibrium solver.

## Installation

Install with pip (any OS):

    pip install pytzer

On Mac or Linux, you can alternatively install from conda-forge:

    conda install conda-forge::pytzer

However, the above will not work on Windows, because jax is not on conda-forge for Windows - so use pip instead.

Once installed, you will need to set the environment variable `JAX_ENABLE_X64=True`.  For example, using conda:

    conda env config vars set JAX_ENABLE_X64=True

## Development status

Pytzer is in beta.  Tests of the accuracy of its parameters and equations are underway, so results may change.  Its API may change, and modules and functions may be added, removed and renamed.  Use at your own peril!

## Citation

A manuscript describing Pytzer is in preparation.  In the meantime, please cite:

> Humphreys, Matthew P. and Schiller, Abigail J. (2024).  Pytzer: the Pitzer model for chemical activities and equilibria in aqueous solutions in Python (beta).  *Zenodo.* [doi:10.5281/zenodo.2637914](https://doi.org/10.5281/zenodo.2637914).

Please report the version you are using.  You can find this in Python with:

```python
import pytzer as pz
pz.hello()
```

## Acknowledgements

Pytzer is maintained by [Dr Matthew P. Humphreys](https://hseao3.group) at the NIOZ Royal Netherlands Institute for Sea Research (Texel, the Netherlands).  Its initial development at the University of East Anglia was funded indirectly by the Natural Environment Research Council (NERC, UK).

Pytzer contains many functions and coefficients representing the effects of different solute interactions on solution properties that have been empirically determined from painstaking experiments and data compilations by hundreds of researchers over the course at least a century.  We have done our best to list the small selection of this enormous body of work brought together here in the [references](refs).

## License

Pytzer is licensed under the [GNU General Public License version 3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html).
