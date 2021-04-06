# Import and export data

`.io` contains functions to import user data ready for use with Pytzer, and export the results of your calculations.

There are two import functions. Which one should you use?

  * Do you know the exact concentration of every individual component of your solution already? If yes, import with `getmols`.
  * Do you have some 'total' concentrations, and you want Pytzer to solve the chemical speciation for you? If yes, import with `gettots`, and then use the [equilibrate](../equilibrate) functions to calculate the concentration of every solute at equilibrium.

Once you have completed the appropriate step above, you can then use the [model](../model) functions to calculate the solution's properties.

---

## `.getmols` - import CSV dataset

Imports a table of temperature, pressure and molality data, formatted ready for Pytzer's [model functions](../model). For calculations without equilibration, where the concentration of every individual solute is specified in the input file.

**Syntax:**

```python
mols, ions, tempK, pres = pz.io.getmols(filename, delimiter=',', skip_top=0)
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

As another example, you could take a look at [pytzerQuickStart.csv](https://raw.githubusercontent.com/mvdh7/pytzer/master/testfiles/pytzerQuickStart.csv).

Note: oceanographers typically record pressure within the ocean as only due to the water, so at the sea surface the pressure would be 0 dbar. However, the atmospheric pressure (1 atm = 10.1325 dbar) should also be taken into account for calculations within Pytzer.

**Outputs:**

  * `mols` - concentrations (molality) of solutes in mol·kg<sup>−1</sup>. Each row represents a different ion, while each column characterises a different solution composition;
  * `ions` - list of the solute codes, corresponding to the rows in `mols`;
  * `tempK` - solution temperature in K. Each value corresponds to the matching column in `mols`;
  * `pres` - solution pressure in dbar. Each value corresponds to the matching column in `mols`.

---

## `.gettots` - import CSV dataset

Imports a table of temperature, pressure and molality data, formatted ready for Pytzer's [equilibration functions](../equilibrate). For calculations with equilibration, where some solution components are given as totals.

**Syntax:**

```python
tots, mols, eles, ions, tempK, pres = pz.io.gettots(filename, delimiter=',', skip_top=0)
```

**Inputs:**

  * `filename` - path and name of a text file containing a set of solution compositions and corresponding temperatures. See full decription below;
  * `delimiter` - column delimiter. Optional; defaults to `','`.
  * `skip_top` - number of rows in `filename` above the row containing the headers described below. Optional; defaults to `0`.

The file should be in comma-separated variable (CSV) format, and its contents follow the same rules as for the `getmols` function above. However, it may also include *total* concentrations for solutes that are in dynamic equilibria to be solved. These are indicated by prefixing the solute name with `t_` in the header row.

**Outputs:**

  * `tots` - total concentrations (molality) of equilibrating solutes in mol·kg<sup>−1</sup>. Each row represents a different set of solutes, while each column characterises a different solution composition;
  * `mols` - concentrations (molality) of non-equilibrating solutes in mol·kg<sup>−1</sup>. Each row represents a different solute, while each column characterises a different solution composition;
  * `eles` - list of the solute codes corresponding to the rows in `tots`;
  * `ions` - list of the solute codes corresponding to the rows in `mols`;
  * `tempK` - solution temperature in K. Each value corresponds to the matching column in `mols`;
  * `pres` - solution pressure in dbar. Each value corresponds to the matching column in `mols`.

As an example, you could take a look at [trisASWequilibrium.csv](https://raw.githubusercontent.com/mvdh7/pytzer/master/testfiles/trisASWequilibrium.csv).

---

## `.saveall` - save to CSV

Saves the results of all calculations to a CSV file, with a format similar to that described for the input file above.

**Syntax:**

```python
pz.io.saveall(filename, mols, ions, tempK, pres, osm, aw, acfs)
```

**Inputs:**

The input `filename` gives the name of the file to save, including the path. The file will be overwritten if it already exists, or created if not. It should end with `.csv`.

The other variables are the inputs `mols`, `ions`, `tempK` and `pres` exactly as created by the [data import function above](#getmols-import-csv-dataset), while `osm`, `aw` and `acfs` are the results of the corresponding [model functions](../model).

---

## Converting between molinity and molality

Oceanographers prefer to report the amount of each solute in a solution as a molinity, that is in units of moles per kilogram of *solution* (i.e. mol/kg-seawater). However, Pytzer requires values in moles per kilogram of H<sub>2</sub>O. We provide a couple of functions to approximately convert values between these two formats: `solution2solvent` and `solvent2solution`.

**Syntax:**

```python
mols, tots = pz.io.solution2solvent(mols, ions, tots, eles)
mols, tots = pz.io.solvent2solution(mols, ions, tots, eles)
```

The former function converts both `mols` and `tots` from molinity to molality, as may be required before running Pytzer calculations to get from typical oceanographic units to those required for the physicochemistry. The latter function does the opposite.

Note that if there are any equilibria to be solved then the conversion is, for now, approximate. This is because the total mass of solutes per kilogram of solvent depends on the final equilibrium speciation in the solution. For now, to calculate the conversion factor, we simply assume that each the equilibrating species in `tots` fully dissociates into whichever ion is dominant in typical seawater.

---

## `.salinity2mols` - molalities from salinity

Estimates the solution composition following the simplified artificial seawater recipe of [MZF93](../../references/#MZF93) at the input salinity (in g/kg-sw).

**Syntax:**

```python
mols, ions, tots, eles = pz.io.salinity2mols(salinity, MgOH=False)
```

**Inputs:**

  * `salinity` - salinity in g-salts/kg-sw;
  * `MgOH` - optional; sets whether or not to allow MgOH<sup>+</sup> ion formation (defaults to `False`).

*The calculation approach currently used here is probably not the best way!*
