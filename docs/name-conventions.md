<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
MathJax.Ajax.config.path["mhchem"] =
  "https://cdnjs.cloudflare.com/ajax/libs/mathjax-mhchem/3.3.2";
MathJax.Hub.Config({TeX: {extensions: ["[mhchem]/mhchem.js"]}});
</script><script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

# Conventions for solute codes

It is necessary to label each solute component in several places within Pytzer:

  1. In your CSV data files that quantify the solution composition;

  1. In the `ions` input to the [Pitzer model functions](../modules/model);

  1. In the [coefficient library](../modules/cflibs) that defines the interaction coefficients;

  1. In the functions defining [solute properties](../modules/properties);

  1. In the names of the [interaction coefficient functions](../modules/coeffs).

The convention in Pytzer is to write out the chemical formula for the relevant solute, including internal stoichiometry, excluding brackets, subscript markers and charges. The following tables list all of the codes used for solutes currently available in Pytzer.

## Elemental ions

The elemental ions are arranged in groups and listed in order of atomic number.

Each transition metal may have multiple possible oxidation states. These are indicated in the code with `j` for each positive charge and `q` for each negative: for example, iron(II) and iron(III) (i.e. $\ce{Fe^2+}$ and $\ce{Fe^3+}$) become `Fejj` and `Fejjj` respectively, while $\ce{V-}$ would become `Vq`. The letters `j` and `q` are used because they do not appear in the symbol for any element.

<table><tr>

<td><strong>Solute name</strong></td>
<td><strong>Formula</strong></td>
<td><strong>Name in Pytzer</strong></td>

</tr><tr><td align="center" colspan="3"><em>Alkali metals</em></td>

</tr><tr><td>Hydrogen</td>  <td>$\ce{H+}$</td>  <td><code>H</code></td>
</tr><tr><td>Lithium</td>   <td>$\ce{Li^+}$</td> <td><code>Li</code></td>
</tr><tr><td>Sodium</td>    <td>$\ce{Na^+}$</td> <td><code>Na</code></td>
</tr><tr><td>Potassium</td> <td>$\ce{K^+}$</td>  <td><code>K</code></td>
</tr><tr><td>Rubidium</td>  <td>$\ce{Rb^+}$</td> <td><code>Rb</code></td>
</tr><tr><td>Caesium</td>   <td>$\ce{Cs^+}$</td> <td><code>Cs</code></td>

</tr><tr><td align="center" colspan="3"><em>Alkaline earth metals</em></td>

</tr><tr><td>Magnesium</td> <td>$\ce{Mg^2+}$</td> <td><code>Mg</code></td>
</tr><tr><td>Calcium</td>   <td>$\ce{Ca^2+}$</td> <td><code>Ca</code></td>
</tr><tr><td>Strontium</td> <td>$\ce{Sr^2+}$</td> <td><code>Sr</code></td>
</tr><tr><td>Barium</td>    <td>$\ce{Ba^2+}$</td> <td><code>Ba</code></td>

</tr><tr><td align="center" colspan="3"><em>Transition metals</em></td>

</tr><tr><td>Manganese(II)</td> <td>$\ce{Mn^2+}$</td> <td><code>Mnjj</code></td>
</tr><tr><td>Iron(II)</td>      <td>$\ce{Fe^2+}$</td> <td><code>Fejj</code></td>
</tr><tr><td>Iron(III)</td>     <td>$\ce{Fe^3+}$</td> <td><code>Fejjj</code></td>
</tr><tr><td>Cobalt(II)</td>    <td>$\ce{Co^2+}$</td> <td><code>Cojj</code></td>
</tr><tr><td>Nickel(II)</td>    <td>$\ce{Ni^2+}$</td> <td><code>Nijj</code></td>
</tr><tr><td>Copper(II)</td>    <td>$\ce{Cu^2+}$</td> <td><code>Cujj</code></td>

</tr><tr><td align="center" colspan="3"><em>Post-transition metals</em></td>

</tr><tr><td>Zinc(II)</td>    <td>$\ce{Zn^2+}$</td> <td><code>Znjj</code></td>
</tr><tr><td>Cadmium(II)</td> <td>$\ce{Cd^2+}$</td> <td><code>Cdjj</code></td>

</tr><tr><td align="center" colspan="3"><em>Halogens</em></td>

</tr><tr><td>Fluoride</td> <td>$\ce{F^−}$</td>  <td><code>F</code></td>
</tr><tr><td>Chloride</td> <td>$\ce{Cl^−}$</td> <td><code>Cl</code></td>
</tr><tr><td>Iodide</td>   <td>$\ce{I^−}$</td> <td><code>I</code></td>

</tr><tr><td align="center" colspan="3"><em>Lanthanides</em></td>

</tr><tr><td>Lanthanum</td> <td>$\ce{La^3+}$</td> <td><code>La</code></td>

</tr></table>

## Other ions

The other ions are listed approximately in order of the atomic number of their most interesting component.

<table><tr>

<td><strong>Solute name</strong></td>
<td><strong>Formula</strong></td>
<td><strong>Name in Pytzer</strong></td>

</tr><tr><td>Borate</td> <td>$\ce{B(OH)4-}$</td> <td><code>BOH4</code></td>
</tr><tr><td>Bicarbonate</td> <td>$\ce{HCO3^−}$</td> <td><code>HCO3</code></td>
</tr><tr><td>Carbonate</td> <td>$\ce{CO3^2−}$</td> <td><code>CO3</code></td>
</tr><tr><td>TrisH<sup>+</sup></td> <td>$\ce{(HOCH2)3CNH3+}$</td> <td><code>trisH</code></td>
</tr><tr><td>Nitrate</td> <td>$\ce{NO3^−}$</td> <td><code>NO3</code></td>
</tr><tr><td>Hydroxide</td> <td>$\ce{OH^−}$</td> <td><code>OH</code></td>
</tr><tr><td>Dihydrogen phosphate</td> <td>$\ce{H2PO4-}$</td> <td><code>H2PO4</code></td>
</tr><tr><td>Thiocyanate</td> <td>$\ce{SCN^−}$</td> <td><code>SCN</code></td>
</tr><tr><td>Bisulfate</td> <td>$\ce{HSO4^−}$</td> <td><code>HSO4</code></td>
</tr><tr><td>Sulfate</td> <td>$\ce{SO4^2−}$</td> <td><code>SO4</code></td>
</tr><tr><td>Thiosulfate</td> <td>$\ce{S2O3-}$</td> <td><code>S2O3</code></td>
</tr><tr><td>Chlorate</td> <td>$\ce{ClO3^−}$</td> <td><code>ClO3</code></td>
</tr><tr><td>Perchlorate</td> <td>$\ce{ClO4^−}$</td> <td><code>ClO4</code></td>
</tr><tr><td>Magnesium hydroxide</td> <td>$\ce{MgOH^+}$</td> <td><code>MgOH</code></td>
</tr><tr><td>Ferrocyanide</td> <td>$\ce{[Fe(CN)6]^4-}$</td> <td><code>FejjCN6</code></td>
</tr><tr><td>Ferricyanide</td> <td>$\ce{[Fe(CN)6]^3-}$</td> <td><code>FejjjCN6</code></td>
</tr><tr><td>Bromate</td> <td>$\ce{BrO3^−}$</td> <td><code>BrO3</code></td>
</tr><tr><td>Iodate</td> <td>$\ce{IO3^−}$</td> <td><code>IO3</code></td>
</tr><tr><td>Uranyl</td> <td>$\ce{UO2^2+}$</td> <td><code>UO2</code></td>

</tr></table>

## Neutral species

Neutral species are referred to as `ions` throughout Pytzer, for simplicity.

<table><tr>

<td><strong>Solute name</strong></td>
<td><strong>Formula</strong></td>
<td><strong>Name in Pytzer</strong></td>

</tr><tr><td>Tris</td> <td>$\ce{(HOCH2)3CNH2}$</td> <td><code>tris</code></td>

</tr></table>


## What actually matters?

For solutes, you could actually use whatever name you like, as long as it was applied consistently throughout the first four items on the list at the top of this page (i.e. in input files, in the `ions` variable, in the coefficient library, and in the solute properties functions). You could rename `Na` (sodium ion) as `Rincewind` in all of these places, and everything should still work fine. Using a matching name in the corresponding interaction coefficient functions would not be essential, but highly recommended.

The codes used for different literature references are for convenience only; they do not affect the program.
