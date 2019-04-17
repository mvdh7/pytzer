# Matrix model

`.matrix` encodes an alternative formulation of the Pitzer model, using matrix mathematics instead of loops to cycle through the different combinations of solutes.

## `.assemble` - prepare for matrix calculation

Generates coefficient matrices needed for the Pitzer model functions in this module.

**Syntax:**

```python
# Recommended:
allmxs = pz.matrix.assemble(ions, tempK, pres, cflib=Seawater)

# Full:
zs, Aosm, b0mx, b1mx, b2mx, C0mx, C1mx, alph1mx, alph2mx, omegamx, \
        thetamxcc, thetamxaa, psimxcca, psimxcaa \
    = pz.matrix.assemble(ions, tempK, pres, cflib=Seawater)
```

**Inputs:**

  * `ions` - list of ions, as described in the [import/export documentation](../io/#getmols-import-csv-dataset);
  * `tempK` - a single temperature in K;
  * `pres` - a single pressure in dbar;
  * `cflib` - a coefficient library (optional, defaults to Seawater).

**Outputs:**

Every matrix required as an input to the model functions in this module.

## Model functions

All other functions in this module are the matrix equivalents of the functions with the same names [in the main model](../model).

**Syntax:**

```python
# Recommended:
var = pz.matrix.var(mols, *allmxs)

# Full:
var = pz.matrix.var(mols, zs, Aosm, b0mx, b1mx, b2mx, C0mx, C1mx,
    alph1mx, alph2mx, omegamx, thetamxcc, thetamxaa, psimxcca, psimxcaa)
```
