# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2024  M.P. Humphreys  (GNU GPLv3)
import jax
from jax import numpy as np

from .. import (
    debyehueckel,
    unsymmetrical,
)
from .. import (
    dissociation as k,
)
from .. import (
    parameters as p,
)
from ..equilibrate import components as c
from . import Library

# Initialise
library = Library(name="CWTD23")
library.update_Aphi(debyehueckel.Aosm_M88)  # From Table S13
library.update_func_J(unsymmetrical.P75_eq47)

# Tables S14-S18 (beta and C coefficients)
library.update_ca("Ca", "Br", p.bC_Ca_Br_SP78)
library.update_ca("Ca", "BOH4", p.bC_Ca_BOH4_SRM87)
library.update_ca("Ca", "Cl", p.bC_Ca_Cl_GM89)  # CWTD23 cite M88 but uses GM89
library.update_ca(
    "Ca", "HCO3", p.bC_Ca_HCO3_CWTD23
)  # CWTD23 cite HM93 but it's not - it's POS85 with a digit missing
library.update_ca("Ca", "HSO4", p.bC_Ca_HSO4_HMW84)
library.update_ca("Ca", "OH", p.bC_Ca_OH_HMW84)
library.update_ca("Ca", "SO4", p.bC_Ca_SO4_HEW82)
library.update_ca("CaF", "Cl", p.bC_CaF_Cl_PM16)
library.update_ca("H", "Br", p.bC_H_Br_MP98)
library.update_ca("H", "Cl", p.bC_H_Cl_CMR93)
library.update_ca("H", "HSO4", p.bC_H_HSO4_CRP94)
library.update_ca("H", "SO4", p.bC_H_SO4_CRP94)
library.update_ca("K", "Br", p.bC_K_Br_CWTD23)
library.update_ca("K", "BOH4", p.bC_K_BOH4_CWTD23)  # CWTD23 cite SRRJ87 but it's not
library.update_ca("K", "Cl", p.bC_K_Cl_GM89)
library.update_ca("K", "CO3", p.bC_K_CO3_CWTD23)  # CWTD23 cite SRG87 but it's not
library.update_ca("K", "F", p.bC_K_F_CWTD23)  # CWTD cite PM73 + SP78
library.update_ca("K", "HCO3", p.bC_K_HCO3_RGW84)
library.update_ca("K", "HSO4", p.bC_K_HSO4_WM13)
library.update_ca("K", "OH", p.bC_K_OH_MP98)
library.update_ca("K", "SO4", p.bC_K_SO4_GM89)
library.update_ca("Mg", "Br", p.bC_Mg_Br_SP78)
library.update_ca("Mg", "BOH4", p.bC_Mg_BOH4_SRM87)  # CWTD23 cite 88SR, numbers agree
library.update_ca("Mg", "Cl", p.bC_Mg_Cl_PP87i)
library.update_ca("Mg", "HCO3", p.bC_Mg_HCO3_CWTD23)  # CWTD23 cite POS85 but it's not
library.update_ca("Mg", "HSO4", p.bC_Mg_HSO4_HMW84)
library.update_ca("Mg", "SO4", p.bC_Mg_SO4_PP86ii)
library.update_ca("MgF", "Cl", p.bC_MgF_Cl_PM16)
library.update_ca("MgOH", "Cl", p.bC_MgOH_Cl_HMW84)
library.update_ca("Na", "Br", p.bC_Na_Br_CWTD23)  # CWTD23 cite 73PM
library.update_ca("Na", "BOH4", p.bC_Na_BOH4_CWTD23)  # TODO check vs MP98 function
library.update_ca("Na", "Cl", p.bC_Na_Cl_M88)
library.update_ca(
    "Na", "CO3", p.bC_Na_CO3_CWTD23b
)  # TODO check code vs table (see functions)
library.update_ca("Na", "F", p.bC_Na_F_CWTD23)
library.update_ca(
    "Na", "HCO3", p.bC_Na_HCO3_CWTD23b
)  # TODO check code vs table (see functions)
library.update_ca("Na", "HSO4", p.bC_Na_HSO4_CWTD23)
library.update_ca("Na", "OH", p.bC_Na_OH_PP87i)
library.update_ca("Na", "SO4", p.bC_Na_SO4_M88)
library.update_ca("Sr", "Br", p.bC_Sr_Br_SP78)
library.update_ca("Sr", "BOH4", p.bC_Ca_BOH4_SRM87)  # CWTD23 use Ca function
library.update_ca("Sr", "Cl", p.bC_Sr_Cl_CWTD23)
library.update_ca("Sr", "HCO3", p.bC_Ca_HCO3_CWTD23)  # CWTD23 use Ca function
library.update_ca("Sr", "HSO4", p.bC_Ca_HSO4_HMW84)  # CWTD23 use Ca function
library.update_ca("Sr", "OH", p.bC_Ca_OH_HMW84)  # CWTD23 use Ca function
library.update_ca("Sr", "SO4", p.bC_Ca_SO4_HEW82)  # CWTD23 use Ca function

# Table S19 (cc theta and psi coefficients)
library.update_cc("Ca", "H", p.theta_Ca_H_RGO81)
library.update_cca("Ca", "H", "Cl", p.psi_Ca_H_Cl_RGO81)
library.update_cc("Ca", "K", p.theta_Ca_K_GM89)
library.update_cca("Ca", "K", "Cl", p.psi_Ca_K_Cl_GM89)
library.update_cc("Ca", "Mg", p.theta_Ca_Mg_HMW84)  # CWTD23 cite HW80
library.update_cca("Ca", "Mg", "Cl", p.psi_Ca_Mg_Cl_HMW84)  # CWTD23 cite HW80
library.update_cca("Ca", "Mg", "SO4", p.psi_Ca_Mg_SO4_HMW84)  # CWTD23 cite HEW82
library.update_cc("Ca", "Na", p.theta_Ca_Na_M88)
library.update_cca("Ca", "Na", "Cl", p.psi_Ca_Na_Cl_M88)
library.update_cca("Ca", "Na", "SO4", p.psi_Ca_Na_SO4_HMW84)
library.update_cc("H", "K", p.theta_H_K_HWT22)  # CWTD23 cite CMR93
library.update_cca("H", "K", "Br", p.psi_H_K_Br_PK74)
library.update_cca("H", "K", "Cl", p.psi_H_K_Cl_HMW84)
library.update_cca("H", "K", "SO4", p.psi_H_K_SO4_HMW84)
library.update_cca("H", "K", "HSO4", p.psi_H_K_HSO4_HMW84)
library.update_cc("H", "Mg", p.theta_H_Mg_RGB80)
library.update_cca("H", "Mg", "Cl", p.psi_H_Mg_Cl_RGB80)
library.update_cca("H", "Mg", "HSO4", p.psi_H_Mg_HSO4_HMW84)  # not cited, but in code
library.update_cc("H", "Na", p.theta_H_Na_HWT22)
library.update_cca("H", "Na", "Br", p.psi_H_Na_Br_PK74)
library.update_cca("H", "Na", "Cl", p.psi_H_Na_Cl_PK74)
library.update_cc("H", "Sr", p.theta_H_Sr_RGRG86)
library.update_cca("H", "Sr", "Cl", p.psi_H_Sr_Cl_RGRG86)
library.update_cca("K", "Mg", "Cl", p.psi_K_Mg_Cl_PP87ii)
library.update_cca("K", "Mg", "SO4", p.psi_K_Mg_SO4_HMW84)  # CWTD23 cite HW80
library.update_cca("K", "Mg", "HSO4", p.psi_K_Mg_HSO4_HMW84)
library.update_cc("K", "Na", p.theta_K_Na_GM89)
library.update_cca("K", "Na", "Br", p.psi_K_Na_Br_PK74)
library.update_cca("K", "Na", "Cl", p.psi_K_Na_Cl_GM89)
library.update_cca("K", "Na", "SO4", p.psi_K_Na_SO4_GM89)
library.update_cc("K", "Sr", p.theta_Na_Sr_MP98)  # CWTD23 use Na-Sr function
library.update_cca("K", "Sr", "Cl", p.psi_Na_Sr_Cl_MP98)  # CWTD23 use Na-Sr function
library.update_cca("Mg", "MgOH", "Cl", p.psi_Mg_MgOH_Cl_HMW84)
library.update_cc("Mg", "Na", p.theta_Mg_Na_HMW84)  # CWTD23 cite P75
library.update_cca("Mg", "Na", "Cl", p.psi_Mg_Na_Cl_PP87ii)
library.update_cca("Mg", "Na", "SO4", p.psi_Mg_Na_SO4_HMW84)  # CWTD23 cite HW80
library.update_cc("Na", "Sr", p.theta_Na_Sr_MP98)
library.update_cca("Na", "Sr", "Cl", p.psi_Na_Sr_Cl_MP98)

# Table S20 (aa theta and psi coefficients)
library.update_aa("BOH4", "Cl", p.theta_BOH4_Cl_CWTD23)
library.update_caa("Ca", "BOH4", "Cl", p.psi_Ca_BOH4_Cl_MP98)  # CWTD23 cite 02P
library.update_caa("Mg", "BOH4", "Cl", p.psi_Mg_BOH4_Cl_MP98)  # CWTD23 cite 02P
library.update_caa("Na", "BOH4", "Cl", p.psi_Na_BOH4_Cl_MP98)  # CWTD23 cite 02P
library.update_aa("BOH4", "SO4", p.theta_BOH4_SO4_FW86)
library.update_aa("Br", "OH", p.theta_Br_OH_PK74)
library.update_caa("K", "Br", "OH", p.psi_K_Br_OH_PK74)
library.update_caa("Na", "Br", "OH", p.psi_Na_Br_OH_PK74)
library.update_aa("CO3", "Cl", p.theta_CO3_Cl_PP82)
library.update_caa("Na", "CO3", "Cl", p.psi_Na_CO3_Cl_TM82)
library.update_aa("Cl", "F", p.theta_Cl_F_MP98)  # CWTD23 cite 88CB
library.update_caa("Na", "Cl", "F", p.psi_Na_Cl_F_CWTD23)  # CWTD23 cite 88CB
library.update_aa("Cl", "HCO3", p.theta_Cl_HCO3_PP82)
library.update_caa("Mg", "Cl", "HCO3", p.psi_Mg_Cl_HCO3_HMW84)
library.update_caa("Na", "Cl", "HCO3", p.psi_Na_Cl_HCO3_PP82)
library.update_aa("Cl", "HSO4", p.theta_Cl_HSO4_HMW84)
library.update_caa("H", "Cl", "HSO4", p.psi_H_Cl_HSO4_HMW84)
library.update_caa("Na", "Cl", "HSO4", p.psi_Na_Cl_HSO4_HMW84)
library.update_aa("Cl", "OH", p.theta_Cl_OH_CWTD23)  # CWTD23 cite 02P
library.update_caa("Ca", "Cl", "OH", p.psi_Ca_Cl_OH_HMW84)
library.update_caa("K", "Cl", "OH", p.psi_K_Cl_OH_HMW84)
library.update_caa("Na", "Cl", "OH", p.psi_Na_Cl_OH_PK74)
library.update_aa("Cl", "SO4", p.theta_Cl_SO4_M88)
library.update_caa("Ca", "Cl", "SO4", p.psi_Ca_Cl_SO4_HMW84)  # CWTD23 cite 82HE/80HW
library.update_caa("K", "Cl", "SO4", p.psi_K_Cl_SO4_GM89)  # CWTD23 cite M88
library.update_caa("Mg", "Cl", "SO4", p.psi_Mg_Cl_SO4_HMW84)  # CWTD23 cite 82HE/80HW
library.update_caa("Na", "Cl", "SO4", p.psi_Na_Cl_SO4_M88)
library.update_aa("CO3", "SO4", p.theta_CO3_SO4_HMW84)
library.update_caa("K", "CO3", "SO4", p.psi_K_CO3_SO4_HMW84)
library.update_caa("Na", "CO3", "SO4", p.psi_Na_CO3_SO4_HMW84)
library.update_aa("HCO3", "SO4", p.theta_HCO3_SO4_HMW84)
library.update_caa("Mg", "HCO3", "SO4", p.psi_Mg_HCO3_SO4_HMW84)
library.update_caa("Na", "HCO3", "SO4", p.psi_Na_HCO3_SO4_HMW84)
library.update_caa("K", "HSO4", "SO4", p.psi_K_HSO4_SO4_HMW84)
library.update_aa("OH", "SO4", p.theta_OH_SO4_HMW84)
library.update_caa("K", "OH", "SO4", p.psi_K_OH_SO4_HMW84)
library.update_caa("Na", "OH", "SO4", p.psi_Na_OH_SO4_HMW84)

# Table S21 (lambda and zeta coefficients)
library.update_na("BOH3", "Cl", p.lambd_BOH3_Cl_FW86)
library.update_nc("BOH3", "K", p.lambd_BOH3_K_FW86)
library.update_nc("BOH3", "Na", p.lambd_BOH3_Na_FW86)
library.update_nca("BOH3", "Na", "SO4", p.zeta_BOH3_Na_SO4_FW86)
library.update_na("BOH3", "SO4", p.lambd_BOH3_SO4_FW86)
library.update_nc("CO2", "Ca", p.lambd_CO2_Ca_HM93)
library.update_nca("CO2", "Ca", "Cl", p.zeta_CO2_Ca_Cl_HM93)
library.update_na("CO2", "Cl", p.lambd_CO2_Cl_HM93)
library.update_nca("CO2", "H", "Cl", p.zeta_CO2_H_Cl_HM93)
library.update_nc("CO2", "K", p.lambd_CO2_K_HM93)
library.update_nca("CO2", "K", "Cl", p.zeta_CO2_K_Cl_HM93)
library.update_nca("CO2", "K", "SO4", p.zeta_CO2_K_SO4_HM93)
library.update_nc("CO2", "Mg", p.lambd_CO2_Mg_HM93)
library.update_nca("CO2", "Mg", "Cl", p.zeta_CO2_Mg_Cl_HM93)
library.update_nca("CO2", "Mg", "SO4", p.zeta_CO2_Mg_SO4_HM93)
library.update_nc("CO2", "Na", p.lambd_CO2_Na_HM93)
library.update_nca("CO2", "Na", "Cl", p.zeta_CO2_Na_Cl_HM93)
library.update_nca("CO2", "Na", "SO4", p.zeta_CO2_Na_SO4_HM93)
library.update_na("CO2", "SO4", p.lambd_CO2_SO4_HM93)
library.update_nc("HF", "Na", p.lambd_HF_Na_MP98)  # CWTD23 cite 88CB
library.update_nc("MgCO3", "Na", p.lambd_MgCO3_Na_CWTD23)  # CWTD23 cite 83MT

# Table S13 (Aphi and equilibria)
library.update_equilibrium("BOH3", k.BOH3_M79)
library.update_equilibrium("CaCO3", k.CaCO3_MP98_MR97)  # CWTD23 cite 84HM + 88PP
library.update_equilibrium("CaF", k.CaF_MP98_MR97)  # CWTF23 cite 82MS
library.update_equilibrium("HCO3", k.HCO3_MP98)  # TODO SOMETHING WEIRD HERE IN CWTF23
library.update_equilibrium("H2CO3", k.H2CO3_MP98)  # CWTF23 cite 79M
library.update_equilibrium("HF", k.HF_MP98)  # CWTF23 cite DR79a
library.update_equilibrium("HSO4", k.HSO4_CRP94)  # TODO sign differences wth CWTF23?
library.update_equilibrium("H2O", k.H2O_M79)
library.update_equilibrium("MgCO3", k.MgCO3_MP98_MR97)  # CWTF23 cite 83MT + 88PP
library.update_equilibrium("MgF", k.MgF_MP98_MR97)  # CWTF23 cite 88CB
library.update_equilibrium("MgOH", k.MgOH_CW91_ln)
library.update_equilibrium("SrCO3", k.SrCO3_CWTF23)

library.solver_targets = ("H", "CO3", "F")
library.totals_all = {
    # Equilibrating
    "BOH3",
    "Ca",
    "CO2",
    "F",
    "Mg",
    "Sr",
    "SO4",
    # Non-equilibrating
    "Br",
    "Cl",
    "K",
    "Na",
}
library.stoich_init = lambda totals: np.array(
    [
        7.0,
        -np.log10(totals["CO2"] / 2),
        -np.log10(totals["F"] / 2),
        # -np.log10(totals["CO2"] / 2) if totals["CO2"] > 0 else 0.0,
        # -np.log10(totals["F"] / 2) if totals["F"] > 0 else 0.0,
    ]
)


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
    co3 = 10 ** -stoich[library.solver_targets.index("CO3")]
    f = 10 ** -stoich[library.solver_targets.index("F")]
    po4 = 0.0  # no phosphate in this model
    # Calculate speciation
    totals = totals.copy()
    totals.update({t: 0.0 for t in library.totals_all if t not in totals})
    solutes = totals.copy()
    solutes["H"] = h
    solutes["OH"] = c.get_OH(h, ks)
    solutes["CO3"] = co3
    solutes["HCO3"] = c.get_HCO3(h, co3, ks)
    solutes["CO2"] = c.get_CO2(h, co3, ks)
    solutes["Ca"] = c.get_Ca(h, f, co3, po4, totals, ks)
    solutes["Mg"] = c.get_Mg(h, f, co3, po4, totals, ks)
    solutes["Sr"] = c.get_Sr(co3, totals, ks)
    solutes["CaCO3"] = c.get_CaCO3(h, f, co3, po4, totals, ks)
    solutes["MgCO3"] = c.get_MgCO3(h, f, co3, po4, totals, ks)
    solutes["SrCO3"] = c.get_SrCO3(co3, totals, ks)
    solutes["BOH3"] = c.get_BOH3(h, totals, ks)
    solutes["BOH4"] = c.get_BOH4(h, totals, ks)
    solutes["HSO4"] = c.get_HSO4(h, totals, ks)
    solutes["SO4"] = c.get_SO4(h, totals, ks)
    solutes["F"] = f
    solutes["HF"] = c.get_HF(h, f, ks)
    solutes["MgOH"] = c.get_MgOH(h, f, co3, po4, totals, ks)
    solutes["CaF"] = c.get_CaF(h, f, co3, po4, totals, ks)
    solutes["MgF"] = c.get_MgF(h, f, co3, po4, totals, ks)
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
    co3 = 10 ** -stoich[library.solver_targets.index("CO3")]
    f = 10 ** -stoich[library.solver_targets.index("F")]
    po4 = 0.0  # no phosphate in this model
    # Calculate components that will be used more than once
    hco3 = c.get_HCO3(h, co3, ks)
    caco3 = c.get_CaCO3(h, f, co3, po4, totals, ks)
    mgco3 = c.get_MgCO3(h, f, co3, po4, totals, ks)
    srco3 = c.get_SrCO3(co3, totals, ks)
    hf = c.get_HF(h, f, ks)
    # Calculate buffer alkalinity
    alkalinity = (
        c.get_OH(h, ks)
        - h
        + hco3
        + 2 * co3
        + 2 * caco3
        + 2 * mgco3
        + 2 * srco3
        + c.get_BOH4(h, totals, ks)
        - c.get_HSO4(h, totals, ks)
        - hf
        + c.get_MgOH(h, f, co3, po4, totals, ks)
        # TODO (how) do MgF and CaF contribute to alkalinity?
    )
    # Calculate other totals
    co2 = c.get_CO2(h, co3, ks)
    total_CO2 = co2 + hco3 + co3 + caco3 + mgco3 + srco3
    total_F = (
        f
        + hf
        + c.get_CaF(h, f, co3, po4, totals, ks)
        + c.get_MgF(h, f, co3, po4, totals, ks)
    )
    return np.array([alkalinity, total_CO2, total_F]) - stoich_targets


get_stoich_error_jac = jax.jit(jax.jacfwd(get_stoich_error))


def get_alkalinity_explicit(totals):
    return (
        totals["Na"]
        + totals["K"]
        - totals["Cl"]
        - totals["Br"]
        + totals["Mg"] * 2
        + totals["Ca"] * 2
        + totals["Sr"] * 2
        - totals["F"]
        - totals["SO4"] * 2
    )


@jax.jit
def get_stoich_targets(totals):
    return np.array(
        [
            get_alkalinity_explicit(totals),
            totals["CO2"],
            totals["F"],
        ]
    )


library.get_ks_constants = get_ks_constants
library.totals_to_solutes = totals_to_solutes
library.get_alkalinity_explicit = get_alkalinity_explicit
library.get_stoich_error = get_stoich_error
library.get_stoich_targets = get_stoich_targets
library.get_stoich_error_jac = get_stoich_error_jac
