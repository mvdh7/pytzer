# Import and export data

**pytzer.io** provides functions to import user data ready for use with **pytzer**, and export the results of your calculations.

<hr />

## .getmols

Imports a table of temperature, pressure and molality values, and format them ready for **pz.model** functions.

### Syntax

```python
mols, ions, tempK, pres = pz.io.getmols(filename, delimiter=',', skip_top=0)
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
<td><em>Optional.</em> Specify the column delimiter used in <code>filename</code>. Defaults to <code>','</code>.</td>

</tr><tr>

<td><code>skip_top</code></td>
<td><em>Optional.</em> Specify the number of rows in <code>filename</code> above the row containing the headers described below. Defaults to <code>0</code>.</td>
</tr></table>

The input file `filename` should be formatted as follows:

  * the first row provides the header for each column;
  * each column represents the concentration of a different ion (or the temperature);
  * each row characterises a different solution composition.

The header of the column containing the temperatures (in K) must be `tempK`, and the pressures (in dbar) must be headed with `pres`.

The other columns should be headed with the chemical symbol for the ion, excluding the charge, and without any brackets. Only *internal* stoichiometry should be included: SO<sub>4</sub><sup>2−</sup> becomes `SO4`; Na<sup>+</sup> becomes `Na`, and would *not* be `Na2` even if a solution of Na<sub>2</sub>SO<sub>4</sub> was under investigation. For a more detailed explanation, see the [section on naming conventions](../../name-conventions).

The order of the columns does not matter.

By default, it should be a comma-separated variable (CSV) file, but if another separator is used, this can be specified with the optional `delimiter` input.

For example, to specify these solutions:

  * NaCl at 0.10, 3.00 and 6.25 mol·kg<sup>−1</sup>,
  * KCl at 0.50 and 3.45 mol·kg<sup>−1</sup>,
  * a mixture containing 1.20 mol·kg<sup>−1</sup> NaCl and 0.80 mol·kg<sup>−1</sup> KCl,
  * Na<sub>2</sub>SO<sub>4</sub> at 2.00 mol·kg<sup>−1</sup>,
  * all at 298.15 K, and 1 atm

we could use this CSV file:

```text
tempK , pres   , Na  , K   , Cl  , SO4
298.15, 10.1325, 0.10,     , 0.10,
298.15, 10.1325, 3.00,     , 3.00,
298.15, 10.1325, 6.25,     , 6.25,
298.15, 10.1325,     , 0.50, 0.50,
298.15, 10.1325,     , 3.45, 3.45,
298.15, 10.1325, 1.20, 0.80, 2.00,
298.15, 10.1325, 4.00,     ,     , 2.00
```

### Outputs

<table><tr>

<td><strong>Variable</strong></td>
<td><strong>Description</strong></td>
<td><strong>Unit</strong></td>

</tr><tr>

<td><code>mols</code></td>
<td>Concentrations (molality) of ions in solution. Each row represents a different ion. Each column characterises a different solution composition.</td>
<td>mol·kg<sup>−1</sup></td>

</tr><tr>

<td><code>ions</code></td>
<td>List of the ions, corresponding to the rows in <code>mols</code>.</td>
<td>mol·kg<sup>−1</sup></td>

</tr><tr>

<td><code>tempK</code></td>
<td>Solution temperature. Each value corresponds to the matching column in <code>mols</code>.</td>
<td>K</td>

</tr><tr>

<td><code>pres</code></td>
<td>Solution pres. Each value corresponds to the matching column in <code>mols</code>.</td>
<td>dbar</td>

</tr></table>

<hr />

## .saveall

Saves the results of all calculations to a CSV file, with a format similar to that described for the input file above.

### Syntax

```python
pz.io.saveall(filename, mols, ions, tempK, pres, osm, aw, acfs)
```

### Inputs

The input `filename` gives the name of the file to save, including the path. The file will be overwritten if it already exists, or created if not. It should end with `.csv`.

The other variables are the inputs `mols`, `ions`, `tempK` and `pres` exactly as created by the **io.getmols** function above, while `osm`, `aw` and `acfs` are the results of the corresponding functions in **model**. *For the current version, there is no flexibility in which calculation results can be provided.*
