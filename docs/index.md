<!--<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML' async></script>-->

# pytzer

**pytzer** is a Python 3.6+ implementation of the Pitzer model for chemical activities in aqueous solutions (Pitzer, 1991).


# Installation

If using conda, first create and activate a new environment with Python 3.6 and numpy 1.15. Then:

```shell
pip install pytzer
```

Other versions are probably fine, but untested. We are using Python 3.6 rather than 3.7 to enable planned integration with MATLAB.

See the [quick-start guide](quick-start) for more detailed instructions and examples.


# Development status

**pytzer** is in beta. Tests of the accuracy of its coefficients and equations are underway, so results may change. API may change and functions may be added or removed. Use at your own peril!


# Modules

<table><tr>

<td><strong>Module</strong></td>
<td><strong>Purpose</strong></td>

</tr><tr>
<td><code>pytzer.io</code></td>
<td><a href="modules/io">Import and export data</a></td>

</tr><tr>
<td><code>pytzer.cfdicts</code></td>
<td><a href="modules/cfdicts">Define combinations of model coefficients to use</a></td>

</tr><tr>
<td><code>pytzer.model</code></td>
<td><a href="modules/model">Implement the Pitzer model</a></td>

</tr><tr>
<td><code>pytzer.coeffs</code></td>
<td><a href="modules/coeffs">Define functions to evaluate model coefficients</a></td>

</tr><tr>
<td><code>pytzer.tables</code></td>
<td><a href="modules/tables">Store tables of coefficient values</a></td>

</tr><tr>
<td><code>pytzer.jfuncs</code></td>
<td><a href="modules/jfuncs">Define unsymmetrical mixing functions</a></td>

</tr><tr>
<td><code>pytzer.props</code></td>
<td><a href="modules/props">Define universal ionic properties</a></td>

</tr><tr>
<td><code>pytzer.constants</code></td>
<td><a href="modules/constants">Define physical constants</a></td>

</tr><tr>
<td><code>pytzer.meta</code></td>
<td><a href="modules/meta">Define package metadata</a></td>

</tr></table>

# Acknowledgements

**pytzer** is maintained by [Dr Matthew P. Humphreys](https://mvdh.xyz) at the Centre for Ocean and Atmospheric Sciences, School of Environmental Sciences, University of East Anglia, Norwich, UK.

Its ongoing development is funded by the [Natural Environment Research Council](https://nerc.ukri.org/) (NERC, UK) through *NSFGEO-NERC: A Thermodynamic Chemical Speciation Model for the Oceans, Seas, and Estuaries* (PI Prof Simon Clegg, [NE/P012361/1](http://gotw.nerc.ac.uk/list_full.asp?pcode=NE%2FP012361%2F1)).

# License

<img src="img/1920px-GPLv3_Logo.svg.png" width="25%" />

The entirety of **pytzer** is licensed under the [GNU General Public License version 3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html).

# References

Pitzer, K. S. (1991). “Ion Interaction Approach: Theory and Data Correlation,” in *Activity Coefficients in Electrolyte Solutions, 2nd Edition*, ed. K. S. Pitzer (CRC Press, Florida, USA), 75–153.
