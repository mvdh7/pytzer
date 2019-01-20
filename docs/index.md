# pytzer

# pytzer.io

Importing and exporting data.

## pytzer.io.getmols

Import table of temperature and molality values, and format ready for `pz.model` functions.

### Syntax

```python
mols, ions, T = pz.io.getmols(filename, delimiter=',')
```

### Inputs



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
