# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
"""Solve for equilibrium."""

from collections import OrderedDict
import jax
import numpy as np
from .. import dissociation
from . import components, stoichiometric, thermodynamic
from ..libraries import Seawater


@jax.jit
def _solve(ptargets, totals, ks_constants):
    ptargets = stoichiometric.solve(totals, ks_constants, ptargets=ptargets)
    solutes = components.get_solutes(totals, ks_constants, ptargets)
    return solutes, ks_constants


def solve_manual(totals, ks_constants, params, log_kt_constants, ptargets=None):
    """Solve for thermodynamic equilibrium to return solute molalities and stoichiometric
    equilibrium constants, having manually evaluated the relevant constants.
    """
    if ptargets is None:
        ptargets = stoichiometric.create_ptargets(totals, ks_constants)
    optresult_thermodynamic = thermodynamic.solve(
        totals, ks_constants, params, log_kt_constants, ptargets=ptargets
    )
    ks_constants = thermodynamic.update_ks_constants(
        ks_constants, optresult_thermodynamic
    )
    solutes, ks_constants = _solve(ptargets, totals, ks_constants)
    return solutes, ks_constants


def solve(
    totals,
    exclude_equilibria=None,
    ks_constants=None,
    ks_only=None,
    library=Seawater,
    pressure=10.10325,
    temperature=298.15,
    verbose=False,
):
    """Solve for thermodynamic equilibrium to return solute molalities and stoichiometric
    equilibrium constants.
    """
    # Make first estimates of all stoichiometric dissociation constants
    ks_constants_pz = dissociation.assemble(
        temperature=temperature,
        totals=totals,
    )
    # Override first estimates with user-provided values, if there are any
    if ks_constants is not None:
        ks_constants_pz.update(ks_constants)
    # Evaluate Pitzer model parameters
    solutes = components.find_solutes(totals, ks_constants_pz)
    params, log_kt_constants = library.get_parameters_equilibria(
        solutes=solutes,
        temperature=temperature,
        pressure=pressure,
        verbose=verbose,
    )
    # Don't solve ks_only constants thermodynamically
    if ks_only is not None:
        for ks in ks_only:
            if ks in log_kt_constants:
                log_kt_constants.pop(ks)
    # Exclude some equilibria altogether
    if exclude_equilibria is not None:
        for eq in exclude_equilibria:
            if eq in ks_constants_pz:
                ks_constants_pz.pop(eq)
            if eq in log_kt_constants:
                log_kt_constants.pop(eq)
    # Solve for thermodynamic equilibrium
    ptargets = stoichiometric.create_ptargets(totals, ks_constants_pz)
    optresult_thermodynamic = thermodynamic.solve(
        totals, ks_constants_pz, params, log_kt_constants, ptargets=ptargets
    )
    # Determine final ks_constants and solution composition
    ks_constants_pz = thermodynamic.update_ks_constants(
        ks_constants_pz, optresult_thermodynamic
    )
    solutes, ks_constants = _solve(ptargets, totals, ks_constants_pz)
    pks_constants = OrderedDict((k, -np.log10(v)) for k, v in ks_constants.items())
    return solutes, pks_constants
