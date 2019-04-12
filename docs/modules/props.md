# Solute properties

*The casual user has no need to explicitly call this module.*

`.props` contains functions that retrieve the universal properties of ions and electrolytes. By *universal*, we mean that these do not change depending on the exact model set-up, unlike the Pitzer model coefficients.


## `.charges` - solute charges

For an input array of "ion" names - following the **pytzer** [naming conventions](../../name-conventions) - returns the charge on each ion, and separate lists of cations, anions, and neutral species.

### Syntax

```python
zs, cations, anions, neutrals = pytzer.props.charges(ions)
```

### Input

<table><tr>

<td><strong>Variable</strong></td>
<td><strong>Description</strong></td>

</tr><tr>

<td><code>ions</code></td>
<td>numpy array of ion (and neutral species) names</td>

</tr></table>

### Outputs

<table><tr>

<td><strong>Variable</strong></td>
<td><strong>Description</strong></td>

</tr><tr>

<td><code>zs</code></td>
<td>Charge on each ion (or zero for neutrals), in the same order as input <code>ions</code></td>

</tr><tr>

<td><code>cations</code></td>
<td>numpy array of cation names</td>

</tr><tr>

<td><code>anions</code></td>
<td>numpy array of anion names</td>

</tr><td>

<td><code>neutrals</code></td>
<td>numpy array of neutral species names</td>

</tr></table>
