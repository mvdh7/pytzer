# pytzer

# pytzer.io

Importing and exporting data.

## pytzer.io.getmols

Import table of temperature and molality values, and format them ready for `pz.model` functions.

### Syntax

```python
mols, ions, T = pz.io.getmols(filename, delimiter=',')
```

### Inputs

<table><tr>

<td><strong>Variable</strong></td>
<td><strong>Description</strong></td>

</tr><tr>

<td><code>filename</code></td>
<td>Name of a text file containing a set of solution compositions and corresponding temperatures. See full decription below.</td>

</tr><tr>

<td><code>delimiter</code></td>
<td><em>Optional.</em> Specify the column delimiter used in <code>filename</code>.</td>

</tr></table>

The input file `filename` should be formatted as follows:

  * the first row provides the header for each column;
  * each column represents the concentration of a different ion (or the temperature);
  * each row characterises a different solution composition.

The header of the column containing the temperatures (in K) must be `temp`.

The other columns should be headed with the chemical symbol for the ion, excluding the charge, and without any brackets. Only *internal* stoichiometry should be included: SO<sub>4</sub><sup>2−</sup> becomes `SO4`; Na<sup>+</sup> becomes `Na`, and would *not* be `Na2` even if a solution of Na<sub>2</sub>SO<sub>4</sub> was under investigation.

The order of the columns does not matter.

By default, it should be a comma-separated variable (CSV) file, but if another separator is used, this can be specified with the optional `delimiter` input.

For example, to specify these solutions:

  * NaCl at 0.10, 3.00 and 6.25 mol·kg<sup>−1</sup>,
  * KCl at 0.50 and 3.45 mol·kg<sup>−1</sup>,
  * a mixture containing 1.20 mol·kg<sup>−1</sup> NaCl and 0.80 mol·kg<sup>−1</sup> KCl,
  * Na<sub>2</sub>SO<sub>4</sub> at 2.00 mol·kg<sup>−1</sup>,
  * all at 298.15 K,

we could use this CSV file:

```text
temp  , Na  , K   , Cl  , SO4
298.15, 0.10,     , 0.10,
298.15, 3.00,     , 3.00,
298.15, 6.25,     , 6.25,
298.15,     , 0.50, 0.50,
298.15,     , 3.45, 3.45,
298.15, 1.20, 0.80, 2.00,
298.15, 4.00,     ,     , 2.00
```

### Outputs

<table><tr>

<td><strong>Variable</strong></td>
<td><strong>Description</strong></td>
<td><strong>Unit</strong></td>

</tr><tr>

<td><code>mols</code></td>
<td>Concentrations (molality) of ions in solution. Each column represents a different ion. Each row characterises a different solution composition.</td>
<td>mol·kg<sup>−1</sup></td>

</tr><tr>

<td><code>ions</code></td>
<td>List of the ions, corresponding to the columns in <code>mols</code>.</td>
<td>mol·kg<sup>−1</sup></td>

</tr><tr>

<td><code>T</code></td>
<td>Solution temperature. Each row corresponds to the matching row in <code>mols</code>.</td>
<td>K</td>

</tr></table>
