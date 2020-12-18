<!--<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML' async></script>-->

# Pytzer

**Pytzer** is a Python implementation of the Pitzer model for chemical activities in aqueous solutions ([P91](references/#P91)).

## Installation

    pip install pytzer

See the [quick-start guide](quick-start) for more detailed instructions and examples.

## Development status

Pytzer is in beta.  Tests of the accuracy of its parameters and equations are underway, so results may change.  Its API may change, and modules and functions may be added, removed and renamed.  Use at your own peril!

## Modules

Most users will only need to make use of a few of Pytzer's modules:

  * `io` - imports and exports data.
  * `model` - implements the Pitzer model without chemical speciation.
  * `equilibrate` - solves for equilibrium.

The remaining modules will only be of interest to more advanced users:

  * `libraries` - defines combinations of model parameters (i.e. *parameter libraries*) for the model.
  * `parameters` - defines interaction parameters as functions of temperature and pressure.
  * `tables` - stores tables of model parameter values.
  * `debyehueckel` - defines functions for Debye-Hückel limiting slopes.
  * `unsymmetrical` - defines unsymmetrical mixing functions.
  * `properties` - defines solute properties (e.g. ionic charges, ions in each electrolyte).
  * `constants` - defines physicochemical constants.
  * `teos10` - calculates properties of pure water.
  * `matrix` - implements an alternative matrix-based Pitzer model, used to solve equilibria.
  * `dissociation` - evaluates thermodynamic equilibrium constants.
  * `potentials` - evaluates standard chemical potentials.
  * `meta` - stores metadata about the Pytzer package.

## Citation

A manuscript describing Pytzer is in preparation for publication.  Please check back here or [get in touch](https://mvdh.xyz/contact) to find out how to cite Pytzer in your work.

## Acknowledgements

Pytzer is maintained by [Dr Matthew P. Humphreys](https://mvdh.xyz) at the NIOZ Royal Netherlands Institute for Sea Research (Texel, the Netherlands).  Its initial development at the University of East Anglia was funded by the Natural Environment Research Council (NERC, UK) through grant NE/P012361/1: *NSFGEO-NERC: A Thermodynamic Chemical Speciation Model for the Oceans, Seas, and Estuaries*.

Pytzer contains myriad functions and coefficients representing the effects of different solute interactions on solution properties that have been empirically determined from painstaking experiments and data compilations by hundreds of researchers over the course at least a century.  The small selection of this enormous body of work that we have brought together here is listed in the [references](references).

## License

Pytzer is licensed under the [GNU General Public License version 3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html).
