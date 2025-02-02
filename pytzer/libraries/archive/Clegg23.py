# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2024  M.P. Humphreys  (GNU GPLv3)
import jax
from jax import numpy as np
from . import ParameterLibrary
from .. import (
    debyehueckel,
    dissociation as k,
    equilibrate,
    parameters as prm,
    unsymmetrical,
)

# Following CWTF23 Supp. Info. part 6
Clegg23 = ParameterLibrary(name="Clegg23")
Clegg23.assign_func_J(unsymmetrical.P75_eq47)  # TODO check this

# Tables S14-S18 (beta and C coefficients)
Clegg23.update_ca("Ca", "Br", prm.bC_Ca_Br_SP78)
Clegg23.update_ca("Ca", "BOH4", prm.bC_Ca_BOH4_SRM87)
Clegg23.update_ca("Ca", "Cl", prm.bC_Ca_Cl_GM89)  # CWTD23 cite M88 but uses GM89
Clegg23.update_ca(
    "Ca", "HCO3", prm.bC_Ca_HCO3_CWTD23
)  # CWTD23 cite HM93 but it's not - it's POS85 with a digit missing
Clegg23.update_ca("Ca", "HSO4", prm.bC_Ca_HSO4_HMW84)
Clegg23.update_ca("Ca", "OH", prm.bC_Ca_OH_HMW84)
Clegg23.update_ca("Ca", "SO4", prm.bC_Ca_SO4_HEW82)
Clegg23.update_ca("CaF", "Cl", prm.bC_CaF_Cl_PM16)
Clegg23.update_ca("H", "Br", prm.bC_H_Br_MP98)
Clegg23.update_ca("H", "Cl", prm.bC_H_Cl_CMR93)
Clegg23.update_ca("H", "HSO4", prm.bC_H_HSO4_CRP94)
Clegg23.update_ca("H", "SO4", prm.bC_H_SO4_CRP94)
Clegg23.update_ca("K", "Br", prm.bC_K_Br_CWTD23)
Clegg23.update_ca("K", "BOH4", prm.bC_K_BOH4_CWTD23)  # CWTD23 cite SRRJ87 but it's not
Clegg23.update_ca("K", "Cl", prm.bC_K_Cl_GM89)
Clegg23.update_ca("K", "CO3", prm.bC_K_CO3_CWTD23)  # CWTD23 cite SRG87 but it's not
Clegg23.update_ca("K", "F", prm.bC_K_F_CWTD23)  # CWTD cite PM73 + SP78
Clegg23.update_ca("K", "HCO3", prm.bC_K_HCO3_RGW84)
Clegg23.update_ca("K", "HSO4", prm.bC_K_HSO4_WM13)
Clegg23.update_ca("K", "OH", prm.bC_K_OH_MP98)
Clegg23.update_ca("K", "SO4", prm.bC_K_SO4_GM89)
Clegg23.update_ca("Mg", "Br", prm.bC_Mg_Br_SP78)
Clegg23.update_ca("Mg", "BOH4", prm.bC_Mg_BOH4_SRM87)  # CWTD23 cite 88SR, numbers agree
Clegg23.update_ca("Mg", "Cl", prm.bC_Mg_Cl_PP87i)
Clegg23.update_ca("Mg", "HCO3", prm.bC_Mg_HCO3_CWTD23)  # CWTD23 cite POS85 but it's not
Clegg23.update_ca("Mg", "HSO4", prm.bC_Mg_HSO4_HMW84)
Clegg23.update_ca("Mg", "SO4", prm.bC_Mg_SO4_PP86ii)
Clegg23.update_ca("MgF", "Cl", prm.bC_MgF_Cl_PM16)
Clegg23.update_ca("MgOH", "Cl", prm.bC_MgOH_Cl_HMW84)
Clegg23.update_ca("Na", "Br", prm.bC_Na_Br_CWTD23)  # CWTD23 cite 73PM
Clegg23.update_ca("Na", "BOH4", prm.bC_Na_BOH4_CWTD23)  # TODO check vs MP98 function
Clegg23.update_ca("Na", "Cl", prm.bC_Na_Cl_M88)
Clegg23.update_ca(
    "Na", "CO3", prm.bC_Na_CO3_CWTD23b
)  # TODO check code vs table (see functions)
Clegg23.update_ca("Na", "F", prm.bC_Na_F_CWTD23)
Clegg23.update_ca(
    "Na", "HCO3", prm.bC_Na_HCO3_CWTD23b
)  # TODO check code vs table (see functions)
Clegg23.update_ca("Na", "HSO4", prm.bC_Na_HSO4_CWTD23)
Clegg23.update_ca("Na", "OH", prm.bC_Na_OH_PP87i)
Clegg23.update_ca("Na", "SO4", prm.bC_Na_SO4_M88)
Clegg23.update_ca("Sr", "Br", prm.bC_Sr_Br_SP78)
Clegg23.update_ca("Sr", "BOH4", prm.bC_Ca_BOH4_SRM87)  # CWTD23 use Ca function
Clegg23.update_ca("Sr", "Cl", prm.bC_Sr_Cl_CWTD23)
Clegg23.update_ca("Sr", "HCO3", prm.bC_Ca_HCO3_CWTD23)  # CWTD23 use Ca function
Clegg23.update_ca("Sr", "HSO4", prm.bC_Ca_HSO4_HMW84)  # CWTD23 use Ca function
Clegg23.update_ca("Sr", "OH", prm.bC_Ca_OH_HMW84)  # CWTD23 use Ca function
Clegg23.update_ca("Sr", "SO4", prm.bC_Ca_SO4_HEW82)  # CWTD23 use Ca function

# Table S19 (cc theta and psi coefficients)
Clegg23.update_cc("Ca", "H", prm.theta_Ca_H_RGO81)
Clegg23.update_cca("Ca", "H", "Cl", prm.psi_Ca_H_Cl_RGO81)
Clegg23.update_cc("Ca", "K", prm.theta_Ca_K_GM89)
Clegg23.update_cca("Ca", "K", "Cl", prm.psi_Ca_K_Cl_GM89)
Clegg23.update_cc("Ca", "Mg", prm.theta_Ca_Mg_HMW84)  # CWTD23 cite HW80
Clegg23.update_cca("Ca", "Mg", "Cl", prm.psi_Ca_Mg_Cl_HMW84)  # CWTD23 cite HW80
Clegg23.update_cca("Ca", "Mg", "SO4", prm.psi_Ca_Mg_SO4_HMW84)  # CWTD23 cite HEW82
Clegg23.update_cc("Ca", "Na", prm.theta_Ca_Na_M88)
Clegg23.update_cca("Ca", "Na", "Cl", prm.psi_Ca_Na_Cl_M88)
Clegg23.update_cca("Ca", "Na", "SO4", prm.psi_Ca_Na_SO4_HMW84)
Clegg23.update_cc("H", "K", prm.theta_H_K_HWT22)  # CWTD23 cite CMR93
Clegg23.update_cca("H", "K", "Br", prm.psi_H_K_Br_PK74)
Clegg23.update_cca("H", "K", "Cl", prm.psi_H_K_Cl_HMW84)
Clegg23.update_cca("H", "K", "SO4", prm.psi_H_K_SO4_HMW84)
Clegg23.update_cca("H", "K", "HSO4", prm.psi_H_K_HSO4_HMW84)
Clegg23.update_cc("H", "Mg", prm.theta_H_Mg_RGB80)
Clegg23.update_cca("H", "Mg", "Cl", prm.psi_H_Mg_Cl_RGB80)
Clegg23.update_cca("H", "Mg", "HSO4", prm.psi_H_Mg_HSO4_HMW84)  # not cited, but in code
Clegg23.update_cc("H", "Na", prm.theta_H_Na_HWT22)
Clegg23.update_cca("H", "Na", "Br", prm.psi_H_Na_Br_PK74)
Clegg23.update_cca("H", "Na", "Cl", prm.psi_H_Na_Cl_PK74)
Clegg23.update_cc("H", "Sr", prm.theta_H_Sr_RGRG86)
Clegg23.update_cca("H", "Sr", "Cl", prm.psi_H_Sr_Cl_RGRG86)
Clegg23.update_cca("K", "Mg", "Cl", prm.psi_K_Mg_Cl_PP87ii)
Clegg23.update_cca("K", "Mg", "SO4", prm.psi_K_Mg_SO4_HMW84)  # CWTD23 cite HW80
Clegg23.update_cca("K", "Mg", "HSO4", prm.psi_K_Mg_HSO4_HMW84)
Clegg23.update_cc("K", "Na", prm.theta_K_Na_GM89)
Clegg23.update_cca("K", "Na", "Br", prm.psi_K_Na_Br_PK74)
Clegg23.update_cca("K", "Na", "Cl", prm.psi_K_Na_Cl_GM89)
Clegg23.update_cca("K", "Na", "SO4", prm.psi_K_Na_SO4_GM89)
Clegg23.update_cc("K", "Sr", prm.theta_Na_Sr_MP98)  # CWTD23 use Na-Sr function
Clegg23.update_cca("K", "Sr", "Cl", prm.psi_Na_Sr_Cl_MP98)  # CWTD23 use Na-Sr function
Clegg23.update_cca("Mg", "MgOH", "Cl", prm.psi_Mg_MgOH_Cl_HMW84)
Clegg23.update_cc("Mg", "Na", prm.theta_Mg_Na_HMW84)  # CWTD23 cite P75
Clegg23.update_cca("Mg", "Na", "Cl", prm.psi_Mg_Na_Cl_PP87ii)
Clegg23.update_cca("Mg", "Na", "SO4", prm.psi_Mg_Na_SO4_HMW84)  # CWTD23 cite HW80
Clegg23.update_cc("Na", "Sr", prm.theta_Na_Sr_MP98)
Clegg23.update_cca("Na", "Sr", "Cl", prm.psi_Na_Sr_Cl_MP98)

# Table S20 (aa theta and psi coefficients)
Clegg23.update_aa("BOH4", "Cl", prm.theta_BOH4_Cl_CWTD23)
Clegg23.update_caa("Ca", "BOH4", "Cl", prm.psi_Ca_BOH4_Cl_MP98)  # CWTD23 cite 02P
Clegg23.update_caa("Mg", "BOH4", "Cl", prm.psi_Mg_BOH4_Cl_MP98)  # CWTD23 cite 02P
Clegg23.update_caa("Na", "BOH4", "Cl", prm.psi_Na_BOH4_Cl_MP98)  # CWTD23 cite 02P
Clegg23.update_aa("BOH4", "SO4", prm.theta_BOH4_SO4_FW86)
Clegg23.update_aa("Br", "OH", prm.theta_Br_OH_PK74)
Clegg23.update_caa("K", "Br", "OH", prm.psi_K_Br_OH_PK74)
Clegg23.update_caa("Na", "Br", "OH", prm.psi_Na_Br_OH_PK74)
Clegg23.update_aa("CO3", "Cl", prm.theta_CO3_Cl_PP82)
Clegg23.update_caa("Na", "CO3", "Cl", prm.psi_Na_CO3_Cl_TM82)
Clegg23.update_aa("Cl", "F", prm.theta_Cl_F_MP98)  # CWTD23 cite 88CB
Clegg23.update_caa("Na", "Cl", "F", prm.psi_Na_Cl_F_CWTD23)  # CWTD23 cite 88CB
Clegg23.update_aa("Cl", "HCO3", prm.theta_Cl_HCO3_PP82)
Clegg23.update_caa("Mg", "Cl", "HCO3", prm.psi_Mg_Cl_HCO3_HMW84)
Clegg23.update_caa("Na", "Cl", "HCO3", prm.psi_Na_Cl_HCO3_PP82)
Clegg23.update_aa("Cl", "HSO4", prm.theta_Cl_HSO4_HMW84)
Clegg23.update_caa("H", "Cl", "HSO4", prm.psi_H_Cl_HSO4_HMW84)
Clegg23.update_caa("Na", "Cl", "HSO4", prm.psi_Na_Cl_HSO4_HMW84)
Clegg23.update_aa("Cl", "OH", prm.theta_Cl_OH_CWTD23)  # CWTD23 cite 02P
Clegg23.update_caa("Ca", "Cl", "OH", prm.psi_Ca_Cl_OH_HMW84)
Clegg23.update_caa("K", "Cl", "OH", prm.psi_K_Cl_OH_HMW84)
Clegg23.update_caa("Na", "Cl", "OH", prm.psi_Na_Cl_OH_PK74)
Clegg23.update_aa("Cl", "SO4", prm.theta_Cl_SO4_M88)
Clegg23.update_caa("Ca", "Cl", "SO4", prm.psi_Ca_Cl_SO4_HMW84)  # CWTD23 cite 82HE/80HW
Clegg23.update_caa("K", "Cl", "SO4", prm.psi_K_Cl_SO4_GM89)  # CWTD23 cite M88
Clegg23.update_caa("Mg", "Cl", "SO4", prm.psi_Mg_Cl_SO4_HMW84)  # CWTD23 cite 82HE/80HW
Clegg23.update_caa("Na", "Cl", "SO4", prm.psi_Na_Cl_SO4_M88)
Clegg23.update_aa("CO3", "SO4", prm.theta_CO3_SO4_HMW84)
Clegg23.update_caa("K", "CO3", "SO4", prm.psi_K_CO3_SO4_HMW84)
Clegg23.update_caa("Na", "CO3", "SO4", prm.psi_Na_CO3_SO4_HMW84)
Clegg23.update_aa("HCO3", "SO4", prm.theta_HCO3_SO4_HMW84)
Clegg23.update_caa("Mg", "HCO3", "SO4", prm.psi_Mg_HCO3_SO4_HMW84)
Clegg23.update_caa("Na", "HCO3", "SO4", prm.psi_Na_HCO3_SO4_HMW84)
Clegg23.update_caa("K", "HSO4", "SO4", prm.psi_K_HSO4_SO4_HMW84)
Clegg23.update_aa("OH", "SO4", prm.theta_OH_SO4_HMW84)
Clegg23.update_caa("K", "OH", "SO4", prm.psi_K_OH_SO4_HMW84)
Clegg23.update_caa("Na", "OH", "SO4", prm.psi_Na_OH_SO4_HMW84)

# Table S21 (lambda and zeta coefficients)
Clegg23.update_na("BOH3", "Cl", prm.lambd_BOH3_Cl_FW86)
Clegg23.update_nc("BOH3", "K", prm.lambd_BOH3_K_FW86)
Clegg23.update_nc("BOH3", "Na", prm.lambd_BOH3_Na_FW86)
Clegg23.update_nca("BOH3", "Na", "SO4", prm.zeta_BOH3_Na_SO4_FW86)
Clegg23.update_na("BOH3", "SO4", prm.lambd_BOH3_SO4_FW86)
Clegg23.update_nc("CO2", "Ca", prm.lambd_CO2_Ca_HM93)
Clegg23.update_nca("CO2", "Ca", "Cl", prm.zeta_CO2_Ca_Cl_HM93)
Clegg23.update_na("CO2", "Cl", prm.lambd_CO2_Cl_HM93)
Clegg23.update_nca("CO2", "H", "Cl", prm.zeta_CO2_H_Cl_HM93)
Clegg23.update_nc("CO2", "K", prm.lambd_CO2_K_HM93)
Clegg23.update_nca("CO2", "K", "Cl", prm.zeta_CO2_K_Cl_HM93)
Clegg23.update_nca("CO2", "K", "SO4", prm.zeta_CO2_K_SO4_HM93)
Clegg23.update_nc("CO2", "Mg", prm.lambd_CO2_Mg_HM93)
Clegg23.update_nca("CO2", "Mg", "Cl", prm.zeta_CO2_Mg_Cl_HM93)
Clegg23.update_nca("CO2", "Mg", "SO4", prm.zeta_CO2_Mg_SO4_HM93)
Clegg23.update_nc("CO2", "Na", prm.lambd_CO2_Na_HM93)
Clegg23.update_nca("CO2", "Na", "Cl", prm.zeta_CO2_Na_Cl_HM93)
Clegg23.update_nca("CO2", "Na", "SO4", prm.zeta_CO2_Na_SO4_HM93)
Clegg23.update_na("CO2", "SO4", prm.lambd_CO2_SO4_HM93)
Clegg23.update_nc("HF", "Na", prm.lambd_HF_Na_MP98)  # CWTD23 cite 88CB
Clegg23.update_nc("MgCO3", "Na", prm.lambd_MgCO3_Na_CWTD23)  # CWTD23 cite 83MT

# Table S13 (Aphi and equilibria)
Clegg23.update_Aphi(debyehueckel.Aosm_M88)
Clegg23.update_equilibrium("BOH3", k.BOH3_M79)
Clegg23.update_equilibrium("CaCO3", k.CaCO3_MP98_MR97)  # CWTD23 cite 84HM + 88PP
Clegg23.update_equilibrium("CaF", k.CaF_MP98_MR97)  # CWTF23 cite 82MS
Clegg23.update_equilibrium("HCO3", k.HCO3_MP98)  # TODO SOMETHING WEIRD HERE IN CWTF23
Clegg23.update_equilibrium("H2CO3", k.H2CO3_MP98)  # CWTF23 cite 79M
Clegg23.update_equilibrium("HF", k.HF_MP98)  # CWTF23 cite DR79a
Clegg23.update_equilibrium("HSO4", k.HSO4_CRP94)  # TODO sign differences wth CWTF23?
Clegg23.update_equilibrium("H2O", k.H2O_M79)
Clegg23.update_equilibrium("MgCO3", k.MgCO3_MP98_MR97)  # CWTF23 cite 83MT + 88PP
Clegg23.update_equilibrium("MgF", k.MgF_MP98_MR97)  # CWTF23 cite 88CB
Clegg23.update_equilibrium("MgOH", k.MgOH_CW91_ln)
Clegg23.update_equilibrium("SrCO3", k.SrCO3_CWTF23)
Clegg23["solver_targets"] = ("H", "CO3", "F")
Clegg23["totals_all"] = {
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
# TODO automate generation of solutes_all
Clegg23["solutes_all"] = {
    "BOH3",
    "BOH4",
    "Br",
    "Ca",
    "CaCO3",
    "CaF",
    "Cl",
    "CO2",
    "CO3",
    "F",
    "H",
    "HCO3",
    "HF",
    "HSO4",
    "K",
    "Mg",
    "MgCO3",
    "MgF",
    "MgOH",
    "MgOH",
    "Na",
    "OH",
    "SO4",
    "Sr",
    "SrCO3",
}
Clegg23.add_zero_cc()
Clegg23.add_zero_aa()
Clegg23["nn"] = {}
Clegg23["nnn"] = {}


# Equilibration functions
@jax.jit
def get_ks_constants(thermo):
    exp_thermo = np.exp(thermo)
    ks = {
        eq: exp_thermo[Clegg23["equilibria_all"].index(eq)]
        for eq in Clegg23["equilibria_all"]
    }
    return ks


@jax.jit
def get_solutes(totals, stoich, thermo):
    ks = get_ks_constants(thermo)
    # Extract and convert stoich
    h = 10 ** -stoich[Clegg23["solver_targets"].index("H")]
    co3 = 10 ** -stoich[Clegg23["solver_targets"].index("CO3")]
    f = 10 ** -stoich[Clegg23["solver_targets"].index("F")]
    po4 = 0.0  # no phosphate in this model
    # Calculate speciation
    c = equilibrate.components
    totals = totals.copy()
    totals.update({t: 0.0 for t in Clegg23["totals_all"] if t not in totals})
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
        eq: exp_thermo[Clegg23["equilibria_all"].index(eq)]
        for eq in Clegg23["equilibria_all"]
    }
    # Extract and convert stoich
    h = 10 ** -stoich[Clegg23["solver_targets"].index("H")]
    co3 = 10 ** -stoich[Clegg23["solver_targets"].index("CO3")]
    f = 10 ** -stoich[Clegg23["solver_targets"].index("F")]
    po4 = 0.0  # no phosphate in this model
    # Calculate components that will be used more than once
    c = equilibrate.components
    hco3 = c.get_HCO3(h, co3, ks)
    caco3 = c.get_CaCO3(h, f, co3, po4, totals, ks)
    mgco3 = c.get_MgCO3(h, f, co3, po4, totals, ks)
    srco3 = c.get_SrCO3(co3, totals, ks)
    hf = c.get_HF(h, f, ks)
    # Calculate alkalinity
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


@jax.jit
def get_stoich_targets(totals):
    return np.array(
        [
            equilibrate.stoichiometric.get_explicit_alkalinity(totals),
            totals["CO2"],
            totals["F"],
        ]
    )


get_stoich_error_jac = jax.jit(jax.jacfwd(get_stoich_error))


@jax.jit
def get_stoich_adjust(stoich, totals, thermo, stoich_targets):
    stoich_error = get_stoich_error(stoich, totals, thermo, stoich_targets)
    stoich_error_jac = get_stoich_error_jac(stoich, totals, thermo, stoich_targets)
    stoich_adjust = np.linalg.solve(-stoich_error_jac, stoich_error)
    stoich_adjust = np.where(
        np.abs(stoich_adjust) > 1, np.sign(stoich_adjust), stoich_adjust
    )
    return stoich_adjust


Clegg23["funcs_eq"] = {
    "ks_constants": get_ks_constants,
    "solutes": get_solutes,
    "stoich_error": get_stoich_error,
    "stoich_targets": get_stoich_targets,
    "stoich_error_jac": get_stoich_error_jac,
    "stoich_adjust": get_stoich_adjust,
}
