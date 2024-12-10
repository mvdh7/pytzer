# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2024  M.P. Humphreys  (GNU GPLv3)
import warnings
from collections import namedtuple

import jax
from jax import numpy as np

from .. import model
from ..model import library
from . import thermodynamic


@jax.jit
def get_stoich_adjust(stoich, totals, thermo, stoich_targets):
    stoich_error = library.get_stoich_error(stoich, totals, thermo, stoich_targets)
    stoich_error_jac = library.get_stoich_error_jac(
        stoich, totals, thermo, stoich_targets
    )
    # We need to use .lstsq here instead of .solve so that it still works when some
    # of stoich are zero concentrations
    stoich_adjust = np.linalg.lstsq(-stoich_error_jac, stoich_error)[0]
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


SolveResult = namedtuple("SolveResult", ["solutes", "lnks_constants"])
SolveResultRaw = namedtuple(
    "SolveResultRaw", ["stoich", "thermo", "stoich_adjust", "thermo_adjust"]
)
SolveStoichResultRaw = namedtuple("SolveStoichResultRaw", ["stoich", "stoich_adjust"])


def _solve(
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
    iter_stoich_per_thermo : int, optional
        How many times to iterate the stoich part of the solver per thermo loop, by
        default 3.
    verbose : bool, optional
        Whether to print solving progress, by default False.
    warn_cutoff : float, optional
        If any of the final rounds of adjustments for stoich or thermo are greater than
        this value, print a convergence warning, by default 1e-8.

    Returns
    -------
    SolveResultRaw
        A named tuple with the fields
            stoich : array-like
                The final values of the molality solver targets.
            thermo : array-like
                The final values of the natural logarithms of the stoichiometric
                equilibrium constants.
            stoich_adjust : array-like
                The final set of iterative adjustments applied to stoich.
            thermo_adjust : array-like
                The final set of iterative adjustments applied to thermo.
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
    return SolveResultRaw(stoich, thermo, stoich_adjust, thermo_adjust)


def solve(
    totals,
    temperature,
    pressure,
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
    iter_thermo : int, optional
        How many times to iterate the thermo part of the solver, by default 6.
    iter_stoich_per_thermo : int, optional
        How many times to iterate the stoich part of the solver per thermo loop, by
        default 3.
    verbose : bool, optional
        Whether to print solving progress, by default False.
    warn_cutoff : float, optional
        If any of the final rounds of adjustments for stoich or thermo are greater than
        this value, print a convergence warning, by default 1e-8.

    Returns
    -------
    SolveResult
        A named tuple with the fields
            solutes : dict
                The molality of each solute in the solution.
            lnks_constants : array-like
                The final values of the natural logarithms of the stoichiometric
                equilibrium constants.
    """
    srr = _solve(
        totals,
        temperature,
        pressure,
        stoich=None,
        thermo=None,
        iter_thermo=iter_thermo,
        iter_stoich_per_thermo=iter_stoich_per_thermo,
        verbose=verbose,
        warn_cutoff=warn_cutoff,
    )
    solutes = library.totals_to_solutes(totals, srr.stoich, srr.thermo)
    lnks_constants = {k: srr.thermo[i] for i, k in enumerate(library.equilibria_all)}
    return SolveResult(solutes, lnks_constants)


def solve_stoich(
    totals,
    temperature,
    pressure,
    thermo,
    stoich=None,
    iter_stoich=10,
    verbose=False,
    warn_cutoff=1e-8,
):
    """Solve for equilibrium given a set of stoichiometric equilibrium constants.

    Parameters
    ----------
    totals : dict
        The total molality of each solute or solute system.
    temperature : float
        Temperature in K.
    pressure : float
        Pressure in dbar.
    thermo : array-like
        The natural logarithms of the stoichiometric equilibrium constants.
    stoich : array-like, optional
        First guess for the molality solver targets.
    iter_stoich : int, optional
        How many times to iterate the solver loop, by default 10.
    verbose : bool, optional
        Whether to print solving progress, by default False.
    warn_cutoff : float, optional
        If any of the final rounds of adjustments are greater than this value, print a
        convergence warning, by default 1e-8.

    Returns
    -------
    SolveStoichResultRaw
        A named tuple with the fields
            stoich : array-like
                The final values of the molality solver targets.
            stoich_adjust : array-like
                The final set of iterative adjustments applied to stoich.
    """
    # Solver targets---known from the start
    totals = totals.copy()
    totals.update({t: 0.0 for t in library.totals_all if t not in totals})
    stoich_targets = library.get_stoich_targets(totals)
    if stoich is None:
        stoich = library.stoich_init(totals)
    # Solve!
    for _s in range(iter_stoich):
        stoich_adjust = get_stoich_adjust(stoich, totals, thermo, stoich_targets)
        if verbose:
            print("STOICH", _s + 1)
            print(stoich_adjust)
        stoich = stoich + stoich_adjust
    if np.any(np.abs(stoich_adjust) > warn_cutoff):
        warnings.warn(
            "Solver did not converge below `warn_cutoff` - "
            + "try increasing `iter_stoich`."
        )
    return SolveStoichResultRaw(stoich, stoich_adjust)


def ks_to_thermo(ks_constants):
    return np.log(
        np.array(
            [
                ks_constants[k] if k in ks_constants else 1.0
                for k in library.equilibria_all
            ]
        )
    )
