# Solute properties

*The casual user has no need to explicitly call this module.*

`.props` contains functions that retrieve the universal properties of ions and electrolytes. By *universal*, we mean that these do not change depending on the exact model set-up, unlike the Pitzer model coefficients.


## `.charges` - solute charges

For an input array of "ion" names - following the Pytzer [naming conventions](../../name-conventions) - returns the charge on each ion, and separate lists of cations, anions, and neutral species.

**Syntax:**

```python
zs, cations, anions, neutrals = pytzer.props.charges(ions)
```

**Input:**

  * `ions` - array of ion (and neutral species) codes (see [conventions](../../name-conventions)).

</tr></table>

**Outputs:**

  * `zs` - the charge on each ion (or zero for neutrals), in the same order as input `ions`;
  * `cations` - array of cation names;
  * `anions` - array of anion names;
  * `neutrals` - array of neutral species names.
