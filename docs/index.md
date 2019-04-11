<!--<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML' async></script>-->

# pytzer

**pytzer** is a Python 3.6+ implementation of the Pitzer model for chemical activities in aqueous solutions (Pitzer, 1991).


# Installation

If using conda, first create and activate a new environment with Python v3.6+, numpy v1.15+ and scipy v1.2+. Other versions are probably fine, but untested. Then:

```shell
pip install pytzer
pip install git+https://github.com/mvdh7/autograd#egg=autograd --upgrade --no-cache-dir
```

The second line above is optional. It upgrades [autograd](https://github.com/HIPS/autograd) to the latest version that has been tested with pytzer, which eliminates some deprecation warnings that may appear when using the relatively old autograd version available from PyPI. You could also switch `mvdh7` in the URL to `HIPS` to get the very latest autograd straight from the horse's mouth.

See the [quick-start guide](quick-start) for more detailed instructions and examples.


# Development status

**pytzer** is in beta. Tests of the accuracy of its coefficients and equations are underway, so results may change. API may change and functions may be added or removed. Use at your own peril!


# Modules

<table><tr>

<td><strong>Module</strong></td>
<td><strong>Purpose</strong></td>

</tr><tr>
<td><code>.io</code></td>
<td><a href="modules/io">Import and export data</a></td>

</tr><tr>
<td><code>.cflibs</code></td>
<td><a href="modules/cflibs">Define combinations of model coefficients to use</a></td>

</tr><tr>
<td><code>.model</code></td>
<td><a href="modules/model">Implement the Pitzer model</a></td>

</tr><tr>
<td><code>.coeffs</code></td>
<td><a href="modules/coeffs">Define functions to evaluate model coefficients</a></td>

</tr><tr>
<td><code>.tables</code></td>
<td><a href="modules/tables">Store tables of coefficient values</a></td>

</tr><tr>
<td><code>.jfuncs</code></td>
<td><a href="modules/jfuncs">Define unsymmetrical mixing functions</a></td>

</tr><tr>
<td><code>.props</code></td>
<td><a href="modules/props">Define universal ionic properties</a></td>

</tr><tr>
<td><code>.constants</code></td>
<td><a href="modules/constants">Define physical constants</a></td>

</tr><tr>
<td><code>.teos10</code></td>
<td><a href="modules/teos10">Properties of pure water</a></td>

</tr><tr>
<td><code>.meta</code></td>
<td><a href="modules/meta">Define package metadata</a></td>

</tr></table>

# Acknowledgements

**pytzer** is maintained by [Dr Matthew P. Humphreys](https://mvdh.xyz) at the Centre for Ocean and Atmospheric Sciences, School of Environmental Sciences, University of East Anglia, Norwich, UK.

Its ongoing development is funded by the [Natural Environment Research Council](https://nerc.ukri.org/) (NERC, UK) through *NSFGEO-NERC: A Thermodynamic Chemical Speciation Model for the Oceans, Seas, and Estuaries* (NE/P012361/1).

# License

<!--<img src="img/1920px-GPLv3_Logo.svg.png" width="25%" />-->

The entirety of **pytzer** is licensed under the [GNU General Public License version 3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html).

# References

Pitzer, K. S. (1991). “Ion Interaction Approach: Theory and Data Correlation,” in *Activity Coefficients in Electrolyte Solutions, 2nd Edition*, ed. K. S. Pitzer (CRC Press, Florida, USA), 75–153.
