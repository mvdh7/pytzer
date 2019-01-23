# Solute names

It is necessary to label each solute component in several places within **pytzer**:

  1. In your input files, for data import using **pytzer.io**;

  1. In the `ions` variable that goes into the **pytzer.model** function;

  1. In the **cfdict** that defines the interaction coefficients;

  1. In the **pytzer.props** functions;

  1. In the names of the interaction coefficient functions in **pytzer.coeffs**.

The convention in **pytzer** is to just write out the chemical formula for the relevant ion, including internal stoichiometry (but not subscripted), and excluding charges. No brackets are used. The following tables list all of the names used for different solutes in the functions that are available by default in **pytzer**:

## Elemental ions

The elemental ions are listed in order of atomic number.

<table><tr>

<td><strong>Solute name</strong></td>
<td><strong>Formula</strong></td>
<td><strong>Name in pytzer</strong></td>

</tr><tr><td>Hydrogen</td>  <td>H<sup>+</sup></td>   <td><code>H</code></td>
</tr><tr><td>Lithium</td>   <td>Li<sup>+</sup></td>  <td><code>Li</code></td>
</tr><tr><td>Sodium</td>    <td>Na<sup>+</sup></td>  <td><code>Na</code></td>
</tr><tr><td>Magnesium</td> <td>Mg<sup>2+</sup></td> <td><code>Mg</code></td>
</tr><tr><td>Chlorine</td>  <td>Cl<sup>−</sup></td>  <td><code>Cl</code></td>
</tr><tr><td>Potassium</td> <td>K<sup>+</sup></td>   <td><code>K</code></td>
</tr><tr><td>Calcium</td>   <td>Ca<sup>2+</sup></td> <td><code>Ca</code></td>
</tr><tr><td>Caesium</td>   <td>Cs<sup>+</sup></td>  <td><code>Cs</code></td>

</tr></table>

## Other ions

The other ions are listed approximately in order of the atomic number of their most interesting component.

<table><tr>

<td><strong>Solute name</strong></td>
<td><strong>Formula</strong></td>
<td><strong>Name in pytzer</strong></td>

</tr><tr><td>Borate</td> <td>B(OH)<sub>4</sub><sup>−</sup></td> <td><code>BOH4</code></td>
</tr><tr><td>Bicarbonate</td> <td>HCO<sub>3</sub><sup>−</sup></td> <td><code>HCO3</code></td>
</tr><tr><td>Carbonate</td> <td>CO<sub>3</sub><sup>2−</sup></td> <td><code>CO3</code></td>
</tr><tr><td>TrisH<sup>+</sup></td> <td>(HOCH<sub>2</sub>)<sub>3</sub>CNH<sub>3</sub><sup>+</sup></td> <td><code>trisH</code></td>
</tr><tr><td>Hydroxide</td> <td>OH<sup>−</sup></td> <td><code>OH</code></td>
</tr><tr><td>Bisulfate</td> <td>HSO<sub>4</sub><sup>−</sup></td> <td><code>HSO4</code></td>
</tr><tr><td>Sulfate</td> <td>SO<sub>4</sub><sup>2−</sup></td> <td><code>SO4</code></td>
</tr><tr><td>Magnesium hydroxide</td> <td>MgOH<sup>+</sup></td> <td><code>MgOH</code></td>

</tr></table>

## Neutral species

Neutral species are referred to as `ions` throughout **pytzer**, for simplicity's sake.

<table><tr>

<td><strong>Solute name</strong></td>
<td><strong>Formula</strong></td>
<td><strong>Name in pytzer</strong></td>

</tr><tr><td>Tris</td> <td>(HOCH<sub>2</sub>)<sub>3</sub>CNH<sub>2</sub></td> <td><code>tris</code></td>

</tr></table>

# Literature references

References from the peer-reviewed literature (or 'sources') are written as the initials of the surname of up to the first four authors, followed by the final two digits of the publication year. Extra bits may be added to distinguish between publications that would end up with the same code. In alphabetical order of the source's code:

<table><tr>

<td><strong>Source</strong></td>
<td><strong>Full citation</strong></td>
</tr><tr>

<td><code>A92ii</code></td>
<td>Archer, D. G. (1992). Thermodynamic Properties of the NaCl + H<sub>2</sub>O System. II. Thermodynamic Properties of NaCl(aq), NaCl·2H<sub>2</sub>(cr), and Phase Equilibria. <i>J. Phys. Chem. Ref. Data</i> 21, 793–829. <a href="https://doi.org/10.1063/1.555915">doi:10.1063/1.555915</a>.</td></tr><tr>

<td><code>A99</code></td>
<td>Archer, D. G. (1999). Thermodynamic Properties of the KCl+H<sub>2</sub>O System. <i>J. Phys. Chem. Ref. Data</i> 28, 1–17. <a href="https://doi.org/10.1063/1.556034">doi:10.1063/1.556034</a>.</td></tr><tr>

<td><code>CMR93</code></td>
<td>Campbell, D. M., Millero, F. J., Roy, R., Roy, L., Lawson, M., Vogel, K. M., et al. (1993). The standard potential for the hydrogen-silver, silver chloride electrode in synthetic seawater. <i>Mar. Chem.</i> 44, 221–233. <a href="https://doi.org/10.1016/0304-4203(93)90204-2">doi:10.1016/0304-4203(93)90204-2</a>.</td></tr><tr>

<td><code>CRP94</code></td>
<td>Clegg, S. L., Rard, J. A., and Pitzer, K. S. (1994). Thermodynamic properties of 0–6 mol kg<sup>–1</sup> aqueous sulfuric acid from 273.15 to 328.15 K. <i>J. Chem. Soc., Faraday Trans.</i> 90, 1875–1894. <a href="https://doi.org/10.1039/FT9949001875">doi:10.1039/FT9949001875</a>.</td></tr><tr>

<td><code>dLP83</code></td>
<td>de Lima, M. C. P., and Pitzer, K. S. (1983). Thermodynamics of saturated electrolyte mixtures of NaCl with Na<sub>2</sub>SO<sub>4</sub> and with MgCl<sub>2</sub>. <i>J. Solution Chem.</i> 12, 187–199. <a href="https://doi.org/10.1007/BF00648056">doi:10.1007/BF00648056</a>.</td></tr><tr>

<td><code>GM89</code></td>
<td>Greenberg, J. P., and Møller, N. (1989). The prediction of mineral solubilities in natural waters: A chemical equilibrium model for the Na-K-Ca-Cl-SO<sub>4</sub>-H<sub>2</sub>O system to high concentration from 0 to 250°C. <i>Geochim. Cosmochim. Acta</i> 53, 2503–2518. <a href="https://doi.org/10.1016/0016-7037(89)90124-5">doi:10.1016/0016-7037(89)90124-5</a>.</td></tr><tr>

<td><code>HM83</code></td>
<td>Holmes, H. F., and Mesmer, R. E. (1983). Thermodynamic properties of aqueous solutions of the alkali metal chlorides to 250 °C. <i>J. Phys. Chem.</i> 87, 1242–1255. <a href="https://doi.org/10.1021/j100230a030">doi:10.1021/j100230a030</a>.</td></tr><tr>

<td><code>HMW84</code></td>
<td>Harvie, C. E., Møller, N., and Weare, J. H. (1984). The prediction of mineral solubilities in natural waters: The Na-K-Mg-Ca-H-Cl-SO<sub>4</sub>-OH-HCO<sub>3</sub>-CO<sub>3</sub>-CO<sub>2</sub>-H<sub>2</sub>O system to high ionic strengths at 25°C. <i>Geochim. Cosmochim. Acta</i> 48, 723–751. <a href="https://doi.org/10.1016/0016-7037(84)90098-X">doi:10.1016/0016-7037(84)90098-X</a>.</td></tr><tr>

<td><code>HM86</code></td>
<td>Holmes, H. F., and Mesmer, R. E. (1986). Thermodynamics of aqueous solutions of the alkali metal sulfates. <i>J. Solution Chem.</i> 15, 495–517. <a href="https://doi.org/10.1007/BF00644892">doi:10.1007/BF00644892</a>.</td></tr><tr>

<td><code>HPR93</code></td>
<td>Hovey, J. K., Pitzer, K. S., and Rard, J. A. (1993). Thermodynamics of Na<sub>2</sub>SO<sub>4</sub>(aq) at temperatures <i>T</i> from 273 K to 373 K and of {(1-<i>y</i>)H<sub>2</sub>SO<sub>4</sub>+<i>y</i>Na<sub>2</sub>SO<sub>4</sub>}(aq) at <i>T</i> = 298.15 K.  <a href="https://doi.org/10.1006/jcht.1993.1016">doi:10.1006/jcht.1993.1016</a>.</td></tr><tr>

<td><code>M88</code></td>
<td>Møller, N. (1988). The prediction of mineral solubilities in natural waters: A chemical equilibrium model for the Na-Ca-Cl-SO<sub>4</sub>-H<sub>2</sub>O system, to high temperature and concentration. <i>Geochim. Cosmochim. Acta</i> 52, 821–837. <a href="https://doi.org/10.1016/0016-7037(88)90354-7">doi:10.1016/0016-7037(88)90354-7</a>.</td></tr><tr>

<td><code>MP98</code></td>
<td>Millero, F. J., and Pierrot, D. (1998). A Chemical Equilibrium Model for Natural Waters. <i>Aquat. Geochem.</i> 4, 153–199. <a href="https://doi.org/10.1023/A:1009656023546">doi:10.1023/A:1009656023546</a>.</td></tr><tr>

<td><code>PP87i</code></td>
<td>Pabalan, R. T., and Pitzer, K. S. (1987). Thermodynamics of NaOH(aq) in hydrothermal solutions. <i>Geochim. Cosmochim. Acta</i> 51, 829–837. <a href="https://doi.org/10.1016/0016-7037(87)90096-2">doi:10.1016/0016-7037(87)90096-2</a>.</td></tr><tr>

<td><code>PP86ii</code></td>
<td>Phutela, R. C., and Pitzer, K. S. (1986). Heat capacity and other thermodynamic properties of aqueous magnesium sulfate to 473 K. <i>J. Phys. Chem.</i> 90, 895–901. <a href="https://doi.org/10.1021/j100277a037">doi:10.1021/j100277a037</a>.</td></tr><tr>

<td><code>RC99</code></td>
<td>Rard, J. A., and Clegg, S. L. (1999). Isopiestic determination of the osmotic and activity coefficients of {<i>z</i>H<sub>2</sub>SO<sub>4</sub>+ (1−<i>z</i>)MgSO<sub>4</sub>}(aq) at <i>T</i> = 298.15 K. II. Results for <i>z</i> = (0.43040, 0.28758, and 0.14399) and analysis with Pitzer's model. <i>J. Chem. Thermodyn.</i> 31, 399–429. <a href="https://doi.org/10.1006/jcht.1998.0461">doi:10.1006/jcht.1998.0461</a>.</td></tr><tr>

<td><code>RM81i</code></td>
<td>Rard, J. A., and Miller, D. G. (1981). Isopiestic Determination of the Osmotic Coefficients of Aqueous Na<sub>2</sub>SO<sub>4</sub>, MgSO<sub>4</sub>, and Na<sub>2</sub>SO<sub>4</sub>-MgSO<sub>4</sub> at 25 °C. <i>J. Chem. Eng. Data</i> 26, 33–38. <a href="https://doi.org/10.1021/je00023a013">doi:10.1021/je00023a013</a>.</td></tr><tr>

<td><code>SRRJ87</code></td>
<td>Simonson, J. M., Roy, R. N., Roy, L. N., and Johnson, D. A. (1987). The thermodynamics of aqueous borate solutions I. Mixtures of boric acid with sodium or potassium borate and chloride. <i>J. Solution Chem.</i> 16, 791–803. <a href="https://doi.org/10.1007/BF00650749">doi:10.1007/BF00650749</a>.</td></tr><tr>

<td><code>WM13</code></td>
<td>Waters, J. F., and Millero, F. J. (2013). The free proton concentration scale for seawater pH. <i>Mar. Chem.</i> 149, 8–22. <a href="https://doi.org/10.1016/j.marchem.2012.11.003">doi:10.1016/j.marchem.2012.11.003</a>.</td></tr><tr>

<td><code>ZD17</code></td>
<td>Zezin, D., and Driesner, T. (2017). Thermodynamic properties of aqueous KCl solution at temperatures to 600 K, pressures to 150 MPa, and concentrations to saturation. <i>Fluid Phase Equilib.</i> 453, 24–39. <a href="https://doi.org/10.1016/j.fluid.2017.09.001">doi:10.1016/j.fluid.2017.09.001</a>.</td>

</tr></table>

# What actually matters?

For solutes, you could actually use whatever name you like, as long as it was applied consistently throughout the first four items on the list at the top of this page (i.e. in input files, in the `ions` variable, in the `cfdict`, and in the **pytzer.props** functions). If it was your heart's desire, you could rename `Na` (sodium ion) as `GentooPenguin` in all of these places, and everything should still work fine. Using a matching name in the corresponding interaction coefficient functions is for convenience only, and is not *required* for **pytzer** to run correctly.

The codes used for different references are for convenience only; they do not affect the program.
