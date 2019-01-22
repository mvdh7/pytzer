title: Overview

# pytzer

**pytzer** is a Python 3.6+ implementation of the Pitzer model for chemical speciation (Pitzer, 1991).

# Modules

<table><tr>

<td><strong>Module</strong></td>
<td><strong>Purpose</strong></td>

</tr><tr>

<td><code>pytzer.io</code></td>
<td>Import and export data</td>

</tr><tr>

<td><code>pytzer.cfdicts</code></td>
<td>Define combinations of model coefficients to use</td>

</tr><tr>

<td><code>pytzer.coeffs</code></td>
<td>Define functions to evaluate model coefficients</td>

</tr><tr>

<td><code>pytzer.jfuncs</code></td>
<td>Define unsymmetrical mixing functions</td>

</tr><tr>

<td><code>pytzer.model</code></td>
<td>Implement the Pitzer model</td>

</tr><tr>

<td><code>pytzer.props</code></td>
<td>Define universal ionic properties</td>

</tr><tr>

<td><code>pytzer.constants</code></td>
<td>Define physical constants</td>

</tr></table>

# Acknowledgements

**pytzer** is maintained by Dr Matthew P. Humphreys at the Centre for Ocean and Atmospheric Sciences, School of Environmental Sciences, University of East Anglia, Norwich, UK.

Its development is funded by the Natural Environment Research Council (NERC, UK) through
*NSFGEO-NERC: A Thermodynamic Chemical Speciation Model for the Oceans, Seas, and Estuaries* ([NE/P012361/1](http://gotw.nerc.ac.uk/list_full.asp?pcode=NE%2FP012361%2F1)).

# License

<img src="img/1920px-GPLv3_Logo.svg.png" width="25%" />

The entirety of **pytzer** is licensed under the [GNU General Public License version 3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html).

# References

Pitzer, K. S. (1991). “Ion Interaction Approach: Theory and Data Correlation,” in *Activity Coefficients in Electrolyte Solutions, 2nd Edition*, ed. K. S. Pitzer (CRC Press, Florida, USA), 75–153.
