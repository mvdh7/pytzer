# Matrix model

*The casual user has no need to explicitly call this module.*

`.matrix` encodes an alternative formulation of the Pitzer model, using matrix mathematics instead of loops to cycle through the different combinations of solutes.

This version of the code runs much faster than the equivalent `.model` functions for a single solution. It therefore more suitable for using in the equilibration solver in `.equilibrate`. However, if some property is being evaluated for many different solutions without the need for equilibrium calculations, then the functions in `.model` will probably run faster.

## `.assemble` - prepare for matrix calculation

Generates parameter matrices needed for the Pitzer model functions in this module.

**Syntax:**

```python
allmxs = pz.matrix.assemble(ions, tempK, pres, prmlib=Seawater)
```

**Inputs:**

  * `ions` - list of ions, as described in the [import/export documentation](../io/#getmols-import-csv-dataset);
  * `tempK` - a single temperature in K;
  * `pres` - a single pressure in dbar;
  * `prmlib` - a parameter library (optional, defaults to **Seawater**).

**Outputs:**

Every matrix required as an input to the model functions in this module.

## Model functions

All other functions in this module will be the matrix equivalents of the functions with the same names [in the main model](../model), although so far only the following have been added:

  * `ln_acfs` - natural logarithm of the solute activity coefficients;
  * `acfs` - solute activity coefficients;
  * `ln_aw` - natural logarithm of the water activity;
  * `aw` - water activity.

**Syntax:**

```python
var = pz.matrix.var(mols, allmxs)
```

The input `allmxs` is generated using `.assemble` (see above).
