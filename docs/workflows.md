# Example workflows

## Black boxes

### No equilibration

```python
import pytzer as pz
mols, ions, tempK, pres, prmlib, Gex_nRT, osm, aw, acfs =  pz.blackbox('pytzerQuickStart.csv')
```

### With equilibration

```python
import pytzer as pz
(allmols, allions, tempK, pres, prmlib, Gex_nRT,
    osm, aw, acfs, eqstates) = pz.blackbox_equilibrate('pytzerQuickStart.csv')
```

## With CSV file import

### No equilibration

```python
import pytzer as pz
mols, ions, tempK, pres = pz.io.getmols('pytzerQuickStart.csv')
# Calculate water activity
aw_Seawater = pz.model.aw(mols, ions, tempK, pres)
# Calculate again, but with MarChemSpec parameter library
aw_MarChemSpec = pz.model.aw(mols, ions, tempK, pres,
    prmlib=pz.libraries.MarChemSpec)
# Now with M88 parameter library, padded with zeros for missing interactions
from copy import deepcopy
myM88 = deepcopy(pz.libraries.M88)
myM88.add_zeros()
aw_myM88 = pz.model.aw(mols, ions, tempK, pres, prmlib=myM88)
```

## With equilibration

```python
import pytzer as pz
tots, fixmols, eles, fixions, tempK, pres = pz.io.gettots('trisASWequilibrium.csv')
# Solve first line of the file
eqstate_guess = [30, 0, 0, 0]
allions = pz.properties.getallions(eles, fixions)
eqstate = pz.equilibrate.solve(eqstate_guess, tots[:, 0], fixmols[:, 0], eles,
    allions, fixions, allmxs, lnkHSO4, lnkH2O, lnktrisH)
```
