<!--<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML' async></script>-->

# Pytzer v0.3.0

**Pytzer** is a Python 3.6+ implementation of the Pitzer model for chemical activities in aqueous solutions [[P91](references/#P91)].


## Installation

If using Conda, first create and activate a new environment with Python v3.6+, NumPy v1.15+ and SciPy v1.2+. Other versions are probably fine, but untested. Activate the environment, and then enter:

```shell
pip install pytzer
pip install git+https://github.com/mvdh7/autograd#egg=autograd --upgrade --no-cache-dir
```

The second line above is strongly recommended, but optional. It upgrades [Autograd](https://github.com/HIPS/autograd) to the latest version that has been tested with Pytzer, which eliminates some deprecation warnings that may appear when using the relatively old Autograd version available from PyPI. You could also switch `mvdh7` in the URL to `HIPS` to get the very latest Autograd straight from the horse's mouth.

See the [quick-start guide](quick-start) for more detailed instructions and examples.


## Development status

Pytzer is in beta. Tests of the accuracy of its coefficients and equations are underway, so results may change. API may change and functions may be added or removed. Use at your own peril!


## Modules

<table><tr>

<td><strong>Module</strong></td>
<td><strong>Purpose</strong></td>

</tr><tr><td><code>.io</code></td>
<td><a href="modules/io">Import and export data</a></td>

</tr><tr><td><code>.model</code></td>
<td><a href="modules/model">Implement the Pitzer model</a></td>

</tr><tr><td><code>.cflibs</code></td>
<td><a href="modules/cflibs">Define combinations of model coefficients to use</a></td>

</tr><tr><td><code>.coeffs</code></td>
<td><a href="modules/coeffs">Define functions to evaluate Pitzer model coefficients</a></td>

</tr><tr><td><code>.tables</code></td>
<td><a href="modules/tables">Store tables of coefficient values</a></td>

</tr><tr><td><code>.jfuncs</code></td>
<td><a href="modules/jfuncs">Define unsymmetrical mixing functions</a></td>

</tr><tr><td><code>.props</code></td>
<td><a href="modules/props">Define universal ionic properties</a></td>

</tr><tr><td><code>.constants</code></td>
<td><a href="modules/constants">Define physical constants</a></td>

</tr><tr><td><code>.teos10</code></td>
<td><a href="modules/teos10">Calculate properties of pure water</a></td>

</tr><tr><td><code>.matrix</code></td>
<td><a href="modules/meta">Alternative matrix-based implementation</a></td>

</tr><tr><td><code>.meta</code></td>
<td><a href="modules/meta">Define package metadata</a></td>

</tr></table>

## Acknowledgements

Pytzer is maintained by [Dr Matthew P. Humphreys](https://mvdh.xyz) at the Centre for Ocean and Atmospheric Sciences, School of Environmental Sciences, University of East Anglia, Norwich, UK.

Its ongoing development is funded by the Natural Environment Research Council (NERC, UK) through grant NE/P012361/1: *NSFGEO-NERC: A Thermodynamic Chemical Speciation Model for the Oceans, Seas, and Estuaries*.

## License

Pytzer is licensed under the [GNU General Public License version 3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html).
