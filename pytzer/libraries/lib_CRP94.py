# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2024  M.P. Humphreys  (GNU GPLv3)
import jax
from jax import numpy as np
from .. import (
    debyehueckel,
    dissociation as k,
    unsymmetrical,
    parameters as p,
)
from ..equilibrate import components as c
from . import Library

library = Library(name="CRP94")
library.update_Aphi(debyehueckel.Aosm_CRP94)
library.update_ca("H", "HSO4", p.bC_H_HSO4_CRP94)
library.update_ca("H", "SO4", p.bC_H_SO4_CRP94)
library.update_aa("HSO4", "SO4", p.theta_HSO4_SO4_CRP94)
library.update_caa("H", "HSO4", "SO4", p.psi_H_HSO4_SO4_CRP94)
library.update_func_J(unsymmetrical.P75_eq47)
library.update_equilibrium("HSO4", k.HSO4_CRP94)

# Equilibrium solver stuff
library.solver_targets = ("H",)
library.totals_all = {"SO4"}
library.stoich_init = lambda totals: np.array([7.0])


# Equilibration functions
@jax.jit
def get_ks_constants(thermo):
    exp_thermo = np.exp(thermo)
    ks = {
        eq: exp_thermo[library.equilibria_all.index(eq)]
        for eq in library.equilibria_all
    }
    return ks


@jax.jit
def totals_to_solutes(totals, stoich, thermo):
    ks = get_ks_constants(thermo)
    # Extract and convert stoich
    h = 10 ** -stoich[library.solver_targets.index("H")]
    co3 = 0.0  # no carbonate in this model
    f = 0.0  # no fluoride in this model
    po4 = 0.0  # no phosphate in this model
    # Calculate speciation
    totals = totals.copy()
    totals.update({t: 0.0 for t in library.totals_all if t not in totals})
    solutes = totals.copy()
    solutes["H"] = h
    solutes["HSO4"] = c.get_HSO4(h, totals, ks)
    solutes["SO4"] = c.get_SO4(h, totals, ks)
    return solutes


@jax.jit
def get_stoich_error(stoich, totals, thermo, stoich_targets):
    # Prepare inputs for calculations
    exp_thermo = np.exp(thermo)
    ks = {
        eq: exp_thermo[library.equilibria_all.index(eq)]
        for eq in library.equilibria_all
    }
    # Extract and convert stoich
    h = 10 ** -stoich[library.solver_targets.index("H")]
    co3 = 0.0  # no carbonate in this model
    f = 0.0  # no fluoride in this model
    po4 = 0.0  # no phosphate in this model
    # Calculate alkalinity
    alkalinity = -h - c.get_HSO4(h, totals, ks)
    return np.array([alkalinity]) - stoich_targets


get_stoich_error_jac = jax.jit(jax.jacfwd(get_stoich_error))


def get_alkalinity_explicit(totals):
    return -2 * totals["SO4"]


@jax.jit
def get_stoich_targets(totals):
    return np.array([get_alkalinity_explicit(totals)])


library.get_ks_constants = get_ks_constants
library.totals_to_solutes = totals_to_solutes
library.get_stoich_error = get_stoich_error
library.get_stoich_targets = get_stoich_targets
library.get_stoich_error_jac = get_stoich_error_jac
