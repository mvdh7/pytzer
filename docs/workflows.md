# Example workflows

The CSV files used in these examples are availble [via GitHub](https://github.com/mvdh7/pytzer/tree/master/testfiles).

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
    osm, aw, acfs, eqstates) = pz.blackbox_equilibrate('trisASWequilibrium.csv')
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
