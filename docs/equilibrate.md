# Solving equilibria

There are two 'layers' of solver in Pytzer: stoichiometric and thermodynamic.

The stoichiometric solver determines the molality of each solute for a given set of total molalities and fixed stoichiometric equilibrium constants.

The thermodynamic solver determines the molality of each solute and stoichiometric equilibrium constants for a given set of total molalities and fixed thermodynamic equilibrium constants.

Both solvers are fully compatible with Jax (JIT and grad).

You can solve equilibria using the following functions.  Lower-level approaches with more fine control are possible, but not yet documented.

## Thermodynamic solver

You can solve a single solution for equilibrium using `pz.solve`:

```python
sr = pz.solve(totals, temperature, pressure)
```

### Arguments

  * `totals` is dict of the total molality of each group of components in the solution.  Non-equilibrating components are included too.  Equilibrating components are grouped as follows:
    * `totals["CO2"]` = sum of all carbonate species.
    * `totals["PO4"]` = sum of all phosphate species.
    * `totals["F"]` = sum of all fluoride species.
    * `totals["SO4"]` = sum of all sulfate species.
    * `totals["BOH3"]` = sum of all borate species.
    * `totals["Mg"]` = sum of all magnesium species.
    * `totals["Ca"]` = sum of all calcium species.
    * `totals["Sr"]` = sum of all strontium species.
  * `temperature` is the temperature in K.
  * `pressure` is the pressure in dbar.

### Results

The result `sr` is a `SolveResult` named tuple containing fields corresponding to the solute molalities at equilibria (`sr.solutes`, dict) and the natural logarithms of the stoichiometric equilibrium constants (`sr.lnks_constants`, dict).
