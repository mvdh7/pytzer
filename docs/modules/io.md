<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
MathJax.Ajax.config.path["mhchem"] =
  "https://cdnjs.cloudflare.com/ajax/libs/mathjax-mhchem/3.3.2";
MathJax.Hub.Config({TeX: {extensions: ["[mhchem]/mhchem.js"]}});
</script><script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

# Import and export data

`.io` contains functions to import user data ready for use with Pytzer, and export the results of your calculations.

<hr />

## `.getmols` - import CSV dataset

Imports a table of temperature, pressure and molality data, formatted ready for Pytzer's [model functions](../model).

**Syntax:**

```python
>>> mols, ions, tempK, pres = pz.io.getmols(filename, delimiter=',', skip_top=0)
```

**Inputs:**

  * `filename` - path and name of a text file containing a set of solution compositions and corresponding temperatures. See full decription below;
  * `delimiter` - column delimiter. Optional; defaults to `','`.
  * `skip_top` - number of rows in `filename` above the row containing the headers described below. Optional; defaults to `0`.


The file should be in comma-separated variable (CSV) format, but if another separator is used this can be specified with `delimiter`. The contents should be formatted as follows:

  * the first row provides the header for each column;
  * each column represents the concentration of a different ion (or the temperature or pressure);
  * each row characterises a different solution composition.

The header of the column containing the temperatures (always in K) must be `tempK`, and the pressures (always in dbar) must be headed with `pres`. The order of the columns does not matter. The other columns should be headed with the chemical symbol for the ion, excluding the charge, and without any brackets. Only *internal* stoichiometry should be included: $\ce{SO4^2-}$ becomes `SO4`; $\ce{Na+}$ becomes `Na`, and would *not* be `Na2` even if a solution of $\ce{Na2SO4}$ was under investigation. For a more detailed explanation, see the [naming conventions](../../name-conventions).

For example, to specify these solutions:

  * $\ce{NaCl}$ at 0.10, 3.00 and 6.25 mol·kg<sup>−1</sup>;
  * $\ce{KCl}$ at 0.50 and 3.45 mol·kg<sup>−1</sup>;
  * a mixture containing 1.20 mol·kg<sup>−1</sup> $\ce{NaCl}$ and 0.80 mol·kg<sup>−1</sup> $\ce{KCl}$;
  * $\ce{Na2SO4}$ at 2.00 mol·kg<sup>−1</sup>;
  * all at 298.15 K, and 1 atm (i.e. 10.1325 dbar);

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

Note: oceanographers typically record pressure within the ocean as only due to the water, so at the sea surface the pressure would be 0 dbar. However, the atmospheric pressure (1 atm = 10.1325 dbar) should also be taken into account for calculations within Pytzer.

**Outputs:**

  * `mols` - concentrations (molality) of solutes in mol·kg<sup>−1</sup>. Each row represents a different ion, while each column characterises a different solution composition;
  * `ions` - list of the solute codes, corresponding to the rows in <code>mols</code>;
  * `tempK` - solution temperature in K. Each value corresponds to the matching column in <code>mols</code>;
  * `pres` - solution pressure in dbar. Each value corresponds to the matching column in <code>mols</code>.

<hr />

## .saveall - save to CSV

Saves the results of all calculations to a CSV file, with a format similar to that described for the input file above.

**Syntax:**

```python
>>> pz.io.saveall(filename, mols, ions, tempK, pres, osm, aw, acfs)
```

**Inputs:**

The input `filename` gives the name of the file to save, including the path. The file will be overwritten if it already exists, or created if not. It should end with `.csv`.

The other variables are the inputs `mols`, `ions`, `tempK` and `pres` exactly as created by the [data import function above](#getmols-import-csv-dataset), while `osm`, `aw` and `acfs` are the results of the corresponding [model functions](../model).
