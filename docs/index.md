<!--<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML' async></script>-->

# Pytzer v0.4.1

**Pytzer** is a Python 3.6+ implementation of the Pitzer model for chemical activities in aqueous solutions [[P91](references/#P91)].

## Installation

If using Conda, first create and activate a new environment with Python v3.6+, NumPy v1.15+ and SciPy v1.2+. Other versions are probably fine, but untested. Activate the environment, and then enter:

    pip install pytzer
    pip install git+https://github.com/mvdh7/autograd#egg=autograd --upgrade --no-cache-dir

The second line above is strongly recommended, but optional. It upgrades [Autograd](https://github.com/HIPS/autograd) to the latest version that has been tested with Pytzer, which eliminates some deprecation warnings that may appear when using the relatively old Autograd version available from PyPI. You could also switch `mvdh7` in the URL to `HIPS` to get the very latest Autograd straight from the horse's mouth.

See the [quick-start guide](quick-start) for more detailed instructions and examples.

## Development status

Pytzer is in beta. Tests of the accuracy of its parameters and equations are underway, so results may change. Its API may change, and modules and functions may be added, removed and renamed. Use at your own peril!

## Modules

Most users will only need to make use of a few of Pytzer's modules:

  * `.io` - imports and exports data;
  * `.model` - implements the Pitzer model without chemical speciation;
  * `.equilibrate` - solves for equilibrium.

The remaining modules will only be of interest to more advanced users:

  * `.libraries` - defines combinations of model parameters (i.e. *parameter libraries*) for the model;
  * `.parameters` - defines interaction parameters as functions of temperature and pressure;
  * `.tables` - stores tables of model parameter values;
  * `.debyehueckel` - defines functions for Debye-Hückel limiting slopes;
  * `.unsymmetrical` - defines unsymmetrical mixing functions;
  * `.properties` - defines solute properties (e.g. ionic charges, ions in each electrolyte);
  * `.constants` - defines physicochemical constants;
  * `.teos10` - calculates properties of pure water;
  * `.matrix` - implements an alternative matrix-based Pitzer model, used to solve equilibria;
  * `.dissociation` - evaluates thermodynamic equilibrium constants;
  * `.potentials` - evaluates standard chemical potentials;
  * `.meta` - stores metadata about the Pytzer package.

## Citation

A manuscript describing Pytzer is in preparation for publication. Please check back here or [get in touch](https://mvdh.xyz/contact) to find out how to cite Pytzer in your work.

## Acknowledgements

Pytzer is maintained by [Dr Matthew P. Humphreys](https://mvdh.xyz) at the Centre for Ocean and Atmospheric Sciences, School of Environmental Sciences, University of East Anglia, Norwich, UK.

Its ongoing development is funded by the Natural Environment Research Council (NERC, UK) through grant NE/P012361/1: *NSFGEO-NERC: A Thermodynamic Chemical Speciation Model for the Oceans, Seas, and Estuaries*.

## License

Pytzer is licensed under the [GNU General Public License version 3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html).
