# Solute properties

*The casual user has no need to explicitly call this module.*

`.properties` contains functions that retrieve the universal properties of ions and electrolytes. By *universal*, we mean that these do not change depending on the exact model set-up, unlike the Pitzer model coefficients.

## `.charges` - solute charges

For an input array of "ion" names - following the Pytzer [naming conventions](../../name-conventions) - returns the charge on each ion, and separate lists of cations, anions, and neutral species.

**Syntax:**

```python
zs, cations, anions, neutrals = pz.properties.charges(ions)
```

**Input:**

  * `ions` - array of ion (and neutral species) codes (see [conventions](../../name-conventions)).

</tr></table>

**Outputs:**

  * `zs` - the charge on each ion (or zero for neutrals), in the same order as input `ions`;
  * `cations` - array of cation names;
  * `anions` - array of anion names;
  * `neutrals` - array of neutral species names.

## `getallions` - get ions in each electrolyte

Determine the component ions in an input list of electrolytes and append to an existing list of ions. Hydrogen and hydroxide ions are also added. The ions in each electrolyte are defined in the `_ele2ions` dict also in this module.

**Syntax:**

```python
allions = getallions(eles, fixions)
```

**Inputs:**

  * `eles` - array of electrolyte codes to find the component ions of;
  * `fixions` - array of ions to add the ions in the electrolytes to.

**Output:**

  * `allions` - array of all ion names.
