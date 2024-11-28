# Solving equilibria

There are two 'layers' of solver in Pytzer: stoichiometric and thermodynamic.

The stoichiometric solver determines the molality of each solute given a set of total molalities and fixed stoichiometric equilibrium constants.  It uses a Newton-Raphson iterative method that is fully compatible with JAX.

You can solve equilibria using the following functions.  Lower-level approaches with more fine control are possible, but not yet documented.

## Solve a single solution

You can solve a single solution for equilibrium using `pz.solve`:

```python
scr = pz.solve(
    totals,
    temperature,
    pressure,
)
```

### Arguments

  * `totals` is dict of the total molality of each group of components in the solution.  Non-equilibrating components are included too.  Equilibrating components are grouped as follows:
    * `totals["CO2"]` = sum of all carbonate species.
    * `totals["PO4"]` = sum of all phosphate species.
    * `totals["F"]` = sum of all fluoride species.
    * `totals["SO4"]` = sum of all sulfate species.
    * `totals["BOH3"]` = sum of all borate species.
    * `totals["NH3"]` = sum of all ammonia species.
    * `totals["H2S"]` = sum of all sulfide species.
    * `totals["NO2"]` = sum of all nitrite species.
    * `totals["H4SiO4"]` = sum of all silicate species.
    * `totals["Mg"]` = sum of all magnesium species.
    * `totals["Ca"]` = sum of all calcium species.
    * `totals["Sr"]` = sum of all strontium species.
  * `temperature` is the temperature in K.
  * `pressure` is the pressure in dbar.

### Results

  * `solutes` is an `OrderedDict` of the molality of each component at thermodynamic equilibrium.
  * `ks_constants` is a `dict` of the stoichiometric equilibrium constants at thermodynamic equilibrium.

## Solve a pandas DataFrame

You can put the `totals` described above into columns of a pandas DataFrame (`df`), add columns for `"temperature"` (in K) and `"pressure"` (in dbar) if needed, and then solve all the rows in the DataFrame with `pz.solve_df`:

```python
pz.solve_df(
    df,
    exclude_equilibria=None,
    inplace=True,
    ks_only=None,
    library=Seawater,
    verbose=False,
)
```
