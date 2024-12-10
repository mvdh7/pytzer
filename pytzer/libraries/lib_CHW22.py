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

# First section is a direct copy of the Humphreys22 library
# ---------------------------------------------------------

# Following Supp. Info. part 6
library = Library(name="CHW22")
library.update_func_J(unsymmetrical.P75_eq47)

# Table S6 (Aphi and equilibria)
library.update_Aphi(debyehueckel.Aosm_M88)
library.update_equilibrium("H2O", k.H2O_MF)
library.update_equilibrium("HSO4", k.HSO4_CRP94)
library.update_equilibrium("MgOH", k.MgOH_CW91_ln)

# Tables S7-S11 (beta and C coefficients)
library.update_ca("Ca", "Cl", p.bC_Ca_Cl_GM89)
library.update_ca("Ca", "HSO4", p.bC_Ca_HSO4_WM13)
library.update_ca("Ca", "OH", p.bC_Ca_OH_HMW84)
library.update_ca("Ca", "SO4", p.bC_Ca_SO4_WM13)
library.update_ca("H", "Cl", p.bC_H_Cl_CMR93)
library.update_ca("H", "HSO4", p.bC_H_HSO4_CRP94)
library.update_ca("H", "SO4", p.bC_H_SO4_CRP94)
library.update_ca("K", "Cl", p.bC_K_Cl_GM89)
library.update_ca("K", "HSO4", p.bC_K_HSO4_WM13)
library.update_ca("K", "OH", p.bC_K_OH_HMW84)
library.update_ca("K", "SO4", p.bC_K_SO4_HM86)
library.update_ca("Mg", "Cl", p.bC_Mg_Cl_dLP83)
library.update_ca("Mg", "HSO4", p.bC_Mg_HSO4_RC99)
library.update_ca("Mg", "SO4", p.bC_Mg_SO4_PP86ii)
library.update_ca("MgOH", "Cl", p.bC_MgOH_Cl_HMW84)
library.update_ca("Na", "Cl", p.bC_Na_Cl_M88)
library.update_ca("Na", "HSO4", p.bC_Na_HSO4_HPR93)
library.update_ca("Na", "OH", p.bC_Na_OH_HWT22)
library.update_ca("Na", "SO4", p.bC_Na_SO4_HM86)

# Table S12 (cc theta and psi coefficients)
library.update_cc("Ca", "H", p.theta_Ca_H_RGO81)
library.update_cca("Ca", "H", "Cl", p.psi_Ca_H_Cl_HMW84)
library.update_cc("Ca", "K", p.theta_Ca_K_HMW84)
library.update_cca("Ca", "K", "Cl", p.psi_Ca_K_Cl_HMW84)
library.update_cc("Ca", "Mg", p.theta_Ca_Mg_HMW84)
library.update_cca("Ca", "Mg", "Cl", p.psi_Ca_Mg_Cl_HMW84)
library.update_cca("Ca", "Mg", "SO4", p.psi_Ca_Mg_SO4_HMW84)
library.update_cc("Ca", "Na", p.theta_Ca_Na_HMW84)
library.update_cca("Ca", "Na", "Cl", p.psi_Ca_Na_Cl_HMW84)
library.update_cca("Ca", "Na", "SO4", p.psi_Ca_Na_SO4_HMW84)
library.update_cc("H", "K", p.theta_H_K_HWT22)
library.update_cca("H", "K", "Cl", p.psi_H_K_Cl_HMW84)
library.update_cca("H", "K", "HSO4", p.psi_H_K_HSO4_HMW84)
library.update_cca("H", "K", "SO4", p.psi_H_K_SO4_HMW84)
library.update_cc("H", "Mg", p.theta_H_Mg_RGB80)
library.update_cca("H", "Mg", "Cl", p.psi_H_Mg_Cl_HMW84)
library.update_cca("H", "Mg", "HSO4", p.psi_H_Mg_HSO4_RC99)
library.update_cca("H", "Mg", "SO4", p.psi_H_Mg_SO4_RC99)
library.update_cc("H", "Na", p.theta_H_Na_HWT22)
library.update_cca("H", "Na", "Cl", p.psi_H_Na_Cl_HMW84)
library.update_cca("H", "Na", "HSO4", p.psi_H_Na_HSO4_HMW84)
library.update_cc("K", "Mg", p.theta_K_Mg_HMW84)
library.update_cca("K", "Mg", "Cl", p.psi_K_Mg_Cl_HMW84)
library.update_cca("K", "Mg", "SO4", p.psi_K_Mg_SO4_HMW84)
library.update_cc("K", "Na", p.theta_K_Na_HMW84)
library.update_cca("K", "Na", "Cl", p.psi_K_Na_Cl_HMW84)
library.update_cca("K", "Na", "SO4", p.psi_K_Na_SO4_HMW84)
library.update_cc("Mg", "MgOH", p.theta_Mg_MgOH_HMW84)
library.update_cca("Mg", "MgOH", "Cl", p.psi_Mg_MgOH_Cl_HMW84)
library.update_cc("Mg", "Na", p.theta_Mg_Na_HMW84)
library.update_cca("Mg", "Na", "Cl", p.psi_Mg_Na_Cl_HMW84)
library.update_cca("Mg", "Na", "SO4", p.psi_Mg_Na_SO4_HMW84)

# Table S13 (aa theta and psi coefficients)
library.update_aa("Cl", "HSO4", p.theta_Cl_HSO4_HMW84)
library.update_caa("Na", "Cl", "HSO4", p.psi_Na_Cl_HSO4_HMW84)
library.update_caa("H", "Cl", "HSO4", p.psi_H_Cl_HSO4_HMW84)
library.update_aa("Cl", "OH", p.theta_Cl_OH_HMW84)
library.update_caa("Ca", "Cl", "OH", p.psi_Ca_Cl_OH_HMW84)
library.update_caa("K", "Cl", "OH", p.psi_K_Cl_OH_HMW84)
library.update_caa("Na", "Cl", "OH", p.psi_Na_Cl_OH_HMW84)
library.update_aa("Cl", "SO4", p.theta_Cl_SO4_HMW84)
library.update_caa("Ca", "Cl", "SO4", p.psi_Ca_Cl_SO4_HMW84)
library.update_caa("Mg", "Cl", "SO4", p.psi_Mg_Cl_SO4_HMW84)
library.update_caa("Na", "Cl", "SO4", p.psi_Na_Cl_SO4_HMW84)
library.update_aa("HSO4", "SO4", p.theta_HSO4_SO4_WM13)
library.update_caa("K", "HSO4", "SO4", p.psi_K_HSO4_SO4_HMW84)
library.update_caa("Mg", "HSO4", "SO4", p.psi_Mg_HSO4_SO4_RC99)
library.update_caa("Na", "HSO4", "SO4", p.psi_Na_HSO4_SO4_HMW84)
library.update_aa("OH", "SO4", p.theta_OH_SO4_HMW84)
library.update_caa("K", "OH", "SO4", p.psi_K_OH_SO4_HMW84)
library.update_caa("Na", "OH", "SO4", p.psi_Na_OH_SO4_HMW84)

# Below here is new things from the CHW22 Supp. Info.
# ---------------------------------------------------

# Table S7 (equilibrium constants)
library.update_equilibrium("trisH", k.trisH_BH61)

# Tables S8-S12 (betas and Cs)
library.update_ca("trisH", "Cl", p.bC_trisH_Cl_CHW22)
library.update_ca("trisH", "SO4", p.bC_trisH_SO4_CHW22)

# Table S15 (lambdas and mu)
library.update_nc("tris", "Ca", p.lambd_tris_Ca_CHW22)
library.update_nc("tris", "K", p.lambd_tris_K_CHW22)
library.update_nc("tris", "Mg", p.lambd_tris_Mg_CHW22)
library.update_nc("tris", "Na", p.lambd_tris_Na_CHW22)
library.update_na("tris", "trisH", p.lambd_tris_trisH_LTA21)
library.update_na("tris", "SO4", p.lambd_tris_SO4_LTA21)
library.update_nn("tris", "tris", p.lambd_tris_tris_LTA21)
library.update_nnn("tris", p.mu_tris_tris_tris_LTA21)

# Equilibrium solver stuff
library.solver_targets = ("H",)
library.totals_all = {
    "Mg",
    "SO4",
    "tris",
    # Non-equilibrating
    "Ca",
    "Cl",
    "K",
    "Na",
}
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
    solutes["OH"] = c.get_OH(h, ks)
    solutes["Mg"] = c.get_Mg(h, f, co3, po4, totals, ks)
    solutes["MgOH"] = c.get_MgOH(h, f, co3, po4, totals, ks)
    solutes["HSO4"] = c.get_HSO4(h, totals, ks)
    solutes["SO4"] = c.get_SO4(h, totals, ks)
    solutes["tris"] = c.get_tris(h, totals, ks)
    solutes["trisH"] = c.get_trisH(h, totals, ks)
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
    # Calculate buffer alkalinity
    alkalinity = (
        c.get_OH(h, ks)
        - h
        - c.get_HSO4(h, totals, ks)
        + c.get_MgOH(h, f, co3, po4, totals, ks)
        + c.get_tris(h, totals, ks)
    )
    return np.array([alkalinity]) - stoich_targets


get_stoich_error_jac = jax.jit(jax.jacfwd(get_stoich_error))


def get_alkalinity_explicit(totals):
    return (
        totals["Na"]
        + totals["K"]
        - totals["Cl"]
        + totals["Mg"] * 2
        + totals["Ca"] * 2
        - totals["SO4"] * 2
        + totals["tris"]
    )


@jax.jit
def get_stoich_targets(totals):
    return np.array([get_alkalinity_explicit(totals)])


library.get_ks_constants = get_ks_constants
library.totals_to_solutes = totals_to_solutes
library.get_stoich_error = get_stoich_error
library.get_stoich_targets = get_stoich_targets
library.get_stoich_error_jac = get_stoich_error_jac
