# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2024  M.P. Humphreys  (GNU GPLv3)
from collections import namedtuple
import warnings
import jax
from jax import numpy as np
from . import thermodynamic
from .. import model
from ..model import library


@jax.jit
def get_stoich_adjust(stoich, totals, thermo, stoich_targets):
    stoich_error = library.get_stoich_error(stoich, totals, thermo, stoich_targets)
    stoich_error_jac = library.get_stoich_error_jac(
        stoich, totals, thermo, stoich_targets
    )
    stoich_adjust = np.linalg.solve(-stoich_error_jac, stoich_error)
    stoich_adjust = np.where(
        np.abs(stoich_adjust) > 1, np.sign(stoich_adjust), stoich_adjust
    )
    return stoich_adjust


@jax.jit
def get_thermo_error(thermo, totals, temperature, pressure, stoich, thermo_targets):
    # Prepare inputs for calculations
    lnks = {
        eq: thermo[library.equilibria_all.index(eq)] for eq in library.equilibria_all
    }
    # Calculate speciation
    solutes = library.totals_to_solutes(totals, stoich, thermo)
    # Calculate solute and water activities
    ln_acfs = model.log_activity_coefficients(solutes, temperature, pressure)
    ln_aw = model.log_activity_water(solutes, temperature, pressure)
    # Calculate what the log(K)s apparently are with these stoich/thermo values
    lnk_error = {
        eq: thermodynamic.reactions_all[eq](
            thermo_targets[library.equilibria_all.index(eq)],
            lnks[eq],
            ln_acfs,
            ln_aw,
        )
        for eq in library.equilibria_all
    }
    thermo_error = np.array([lnk_error[eq] for eq in library.equilibria_all])
    return thermo_error


get_thermo_error_jac = jax.jit(jax.jacfwd(get_thermo_error))


@jax.jit
def get_thermo_adjust(thermo, totals, temperature, pressure, stoich, thermo_targets):
    thermo_error = get_thermo_error(
        thermo, totals, temperature, pressure, stoich, thermo_targets
    )
    thermo_error_jac = np.array(
        get_thermo_error_jac(
            thermo, totals, temperature, pressure, stoich, thermo_targets
        )
    )
    thermo_adjust = np.linalg.solve(-thermo_error_jac, thermo_error)
    return thermo_adjust


SolveResult = namedtuple(
    "SolveResult", ["stoich", "thermo", "stoich_adjust", "thermo_adjust"]
)


def solve(
    totals,
    temperature,
    pressure,
    stoich=None,
    thermo=None,
    iter_thermo=6,
    iter_stoich_per_thermo=3,
    verbose=False,
    warn_cutoff=1e-8,
):
    """Solve for equilibrium.

    This function is not JIT-ed by default, because it takes a lot longer to compile,
    but you can JIT it yourself to get a ~2x speedup.

    Parameters
    ----------
    totals : dict
        The total molality of each solute or solute system.
    temperature : float
        Temperature in K.
    pressure : float
        Pressure in dbar.
    stoich : array-like, optional
        First guess for the molality solver targets.
    thermo : array-like, optional
        First guesses for the natural logarithms of the stoichiometric equilibrium
        constants.
    iter_thermo : int, optional
        How many times to iterate the thermo part of the solver, by default 6.
    iter_pH_per_thermo : int, optional
        How many times to iterate the stoich part of the solver per thermo loop, by
        default 3.
    verbose : bool, optional
        Whether to print solving progress, by default False.
    warn_cutoff : float, optional
        If any of the final rounds of adjustments for stoich or thermo are greater than
        this value, print a convergence warning.

    Returns
    -------
    stoich : float
        Final pH.
    thermo : array-like
        Final natural logarithms of the stoichiometric equilibrium constants.
    """
    # Solver targets---known from the start
    totals = totals.copy()
    totals.update({t: 0.0 for t in library.totals_all if t not in totals})
    stoich_targets = library.get_stoich_targets(totals)
    thermo_targets = np.array(
        [library.equilibria[eq](temperature) for eq in library.equilibria_all]
    )  # these are ln(k)
    if stoich is None:
        stoich = library.stoich_init(totals)
    if thermo is None:
        thermo = thermo_targets.copy()
    # Solve!
    for _t in range(iter_thermo):
        for _s in range(iter_stoich_per_thermo):
            stoich_adjust = get_stoich_adjust(stoich, totals, thermo, stoich_targets)
            if verbose:
                print("STOICH", _t + 1, _s + 1)
                print(stoich_adjust)
            stoich = stoich + stoich_adjust
        thermo_adjust = get_thermo_adjust(
            thermo, totals, temperature, pressure, stoich, thermo_targets
        )
        if verbose:
            print("THERMO", _t + 1)
            print(thermo_adjust)
        thermo = thermo + thermo_adjust
    if np.any(np.abs(stoich_adjust) > warn_cutoff):
        warnings.warn(
            "Solver did not converge below `warn_cutoff` - "
            + "try increasing `iter_stoich_per_thermo`."
        )
    if np.any(np.abs(thermo_adjust) > warn_cutoff):
        warnings.warn(
            "Solver did not converge below `warn_cutoff` - "
            + "try increasing `iter_thermo`."
        )
    return SolveResult(stoich, thermo, stoich_adjust, thermo_adjust)
