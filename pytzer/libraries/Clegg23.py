# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2023  Matthew P. Humphreys  (GNU GPLv3)
from . import ParameterLibrary
from .. import debyehueckel, dissociation as k, parameters as prm, unsymmetrical

# Following CWTF23 Supp. Info. part 6
Clegg23 = ParameterLibrary(name="Clegg23")
Clegg23.assign_func_J(unsymmetrical.P75_eq47)  # TODO check this

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

# Tables S14-S18 (beta and C coefficients)
Clegg23.update_ca("Ca", "Br", prm.bC_Ca_Br_SP78)
Clegg23.update_ca("Ca", "BOH4", prm.bC_Ca_BOH4_SRM87)
Clegg23.update_ca("Ca", "Cl", prm.bC_Ca_Cl_GM89)  # CWTD23 cite M88 but uses GM89
Clegg23.update_ca("Ca", "HCO3", prm.bC_Ca_HCO3_HM93)
Clegg23.update_ca("Ca", "HSO4", prm.bC_Ca_HSO4_HMW84)
Clegg23.update_ca("Ca", "OH", prm.bC_Ca_OH_HMW84)
Clegg23.update_ca("Ca", "SO4", prm.bC_Ca_SO4_HEW82)
Clegg23.update_ca("CaF", "Cl", prm.bC_CaF_Cl_PM16)
Clegg23.update_ca("H", "Br", prm.bC_H_Br_MP98)
Clegg23.update_ca("H", "Cl", prm.bC_H_Cl_CMR93)
Clegg23.update_ca("H", "HSO4", prm.bC_H_HSO4_CRP94)
Clegg23.update_ca("H", "SO4", prm.bC_H_SO4_CRP94)
Clegg23.update_ca("K", "Br", prm.bC_K_Br_CWTD23)
Clegg23.update_ca("K", "BOH4", prm.bC_K_BOH4_SRRJ87)
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
Clegg23.update_ca("Mg", "HCO3", prm.bC_Mg_HCO3_POS85)
Clegg23.update_ca("Mg", "HSO4", prm.bC_Mg_HSO4_HMW84)
Clegg23.update_ca("Mg", "SO4", prm.bC_Mg_SO4_PP86ii)
Clegg23.update_ca("MgF", "Cl", prm.bC_MgF_Cl_PM16)
Clegg23.update_ca("MgOH", "Cl", prm.bC_MgOH_Cl_HMW84)
Clegg23.update_ca("Na", "Br", prm.bC_Na_Br_MP98)  # CWTD23 cite 73PM, numbers agree
Clegg23.update_ca("Na", "BOH4", prm.bC_Na_BOH4_SRRJ87)  # TODO check vs MP98 function
Clegg23.update_ca("Na", "Cl", prm.bC_Na_Cl_M88)
Clegg23.update_ca("Na", "CO3", prm.bC_Na_CO3_CWTD23)
Clegg23.update_ca("Na", "F", prm.bC_Na_F_CWTD23)
Clegg23.update_ca("Na", "HCO3", prm.bC_Na_HCO3_CWTD23)
Clegg23.update_ca("Na", "HSO4", prm.bC_Na_HSO4_CWTD23)
Clegg23.update_ca("Na", "OH", prm.bC_Na_OH_PP87i)
Clegg23.update_ca("Na", "SO4", prm.bC_Na_SO4_M88)
Clegg23.update_ca("Sr", "Br", prm.bC_Sr_Br_SP78)
Clegg23.update_ca("Sr", "BOH4", prm.bC_Ca_BOH4_SRM87)  # CWTD23 use Ca function
Clegg23.update_ca("Sr", "Cl", prm.bC_Sr_Cl_CWTD23)
Clegg23.update_ca("Sr", "HCO3", prm.bC_Ca_HCO3_HM93)  # CWTD23 use Ca function
Clegg23.update_ca("Sr", "HSO4", prm.bC_Ca_HSO4_HMW84)  # CWTD23 use Ca function
Clegg23.update_ca("Sr", "OH", prm.bC_Ca_OH_HMW84)  # CWTD23 use Ca function
Clegg23.update_ca("Sr", "SO4", prm.bC_Ca_SO4_HEW82)  # CWTD23 use Ca function


# ALL GOOD ABOVE HERE 2023-10-20 --- still need to do below! ===========================


# Table S12 (cc theta and psi coefficients)
Clegg23.update_cc("Ca", "H", prm.theta_Ca_H_RGO81)
Clegg23.update_cca("Ca", "H", "Cl", prm.psi_Ca_H_Cl_HMW84)
Clegg23.update_cc("Ca", "K", prm.theta_Ca_K_HMW84)
Clegg23.update_cca("Ca", "K", "Cl", prm.psi_Ca_K_Cl_HMW84)
Clegg23.update_cc("Ca", "Mg", prm.theta_Ca_Mg_HMW84)
Clegg23.update_cca("Ca", "Mg", "Cl", prm.psi_Ca_Mg_Cl_HMW84)
Clegg23.update_cca("Ca", "Mg", "SO4", prm.psi_Ca_Mg_SO4_HMW84)
Clegg23.update_cc("Ca", "Na", prm.theta_Ca_Na_HMW84)
Clegg23.update_cca("Ca", "Na", "Cl", prm.psi_Ca_Na_Cl_HMW84)
Clegg23.update_cca("Ca", "Na", "SO4", prm.psi_Ca_Na_SO4_HMW84)
Clegg23.update_cc("H", "K", prm.theta_H_K_HWT22)
Clegg23.update_cca("H", "K", "Cl", prm.psi_H_K_Cl_HMW84)
Clegg23.update_cca("H", "K", "HSO4", prm.psi_H_K_HSO4_HMW84)
Clegg23.update_cca("H", "K", "SO4", prm.psi_H_K_SO4_HMW84)
Clegg23.update_cc("H", "Mg", prm.theta_H_Mg_RGB80)
Clegg23.update_cca("H", "Mg", "Cl", prm.psi_H_Mg_Cl_HMW84)
Clegg23.update_cca("H", "Mg", "HSO4", prm.psi_H_Mg_HSO4_RC99)
Clegg23.update_cca("H", "Mg", "SO4", prm.psi_H_Mg_SO4_RC99)
Clegg23.update_cc("H", "Na", prm.theta_H_Na_HWT22)
Clegg23.update_cca("H", "Na", "Cl", prm.psi_H_Na_Cl_HMW84)
Clegg23.update_cca("H", "Na", "HSO4", prm.psi_H_Na_HSO4_HMW84)
Clegg23.update_cc("K", "Mg", prm.theta_K_Mg_HMW84)
Clegg23.update_cca("K", "Mg", "Cl", prm.psi_K_Mg_Cl_HMW84)
Clegg23.update_cca("K", "Mg", "SO4", prm.psi_K_Mg_SO4_HMW84)
Clegg23.update_cc("K", "Na", prm.theta_K_Na_HMW84)
Clegg23.update_cca("K", "Na", "Cl", prm.psi_K_Na_Cl_HMW84)
Clegg23.update_cca("K", "Na", "SO4", prm.psi_K_Na_SO4_HMW84)
Clegg23.update_cc("Mg", "MgOH", prm.theta_Mg_MgOH_HMW84)
Clegg23.update_cca("Mg", "MgOH", "Cl", prm.psi_Mg_MgOH_Cl_HMW84)
Clegg23.update_cc("Mg", "Na", prm.theta_Mg_Na_HMW84)
Clegg23.update_cca("Mg", "Na", "Cl", prm.psi_Mg_Na_Cl_HMW84)
Clegg23.update_cca("Mg", "Na", "SO4", prm.psi_Mg_Na_SO4_HMW84)

# Table S13 (aa theta and psi coefficients)
Clegg23.update_aa("Cl", "HSO4", prm.theta_Cl_HSO4_HMW84)
Clegg23.update_caa("Na", "Cl", "HSO4", prm.psi_Na_Cl_HSO4_HMW84)
Clegg23.update_caa("H", "Cl", "HSO4", prm.psi_H_Cl_HSO4_HMW84)
Clegg23.update_aa("Cl", "OH", prm.theta_Cl_OH_HMW84)
Clegg23.update_caa("Ca", "Cl", "OH", prm.psi_Ca_Cl_OH_HMW84)
Clegg23.update_caa("K", "Cl", "OH", prm.psi_K_Cl_OH_HMW84)
Clegg23.update_caa("Na", "Cl", "OH", prm.psi_Na_Cl_OH_HMW84)
Clegg23.update_aa("Cl", "SO4", prm.theta_Cl_SO4_HMW84)
Clegg23.update_caa("Ca", "Cl", "SO4", prm.psi_Ca_Cl_SO4_HMW84)
Clegg23.update_caa("Mg", "Cl", "SO4", prm.psi_Mg_Cl_SO4_HMW84)
Clegg23.update_caa("Na", "Cl", "SO4", prm.psi_Na_Cl_SO4_HMW84)
Clegg23.update_aa("HSO4", "SO4", prm.theta_HSO4_SO4_WM13)
Clegg23.update_caa("K", "HSO4", "SO4", prm.psi_K_HSO4_SO4_HMW84)
Clegg23.update_caa("Mg", "HSO4", "SO4", prm.psi_Mg_HSO4_SO4_RC99)
Clegg23.update_caa("Na", "HSO4", "SO4", prm.psi_Na_HSO4_SO4_HMW84)
Clegg23.update_aa("OH", "SO4", prm.theta_OH_SO4_HMW84)
Clegg23.update_caa("K", "OH", "SO4", prm.psi_K_OH_SO4_HMW84)
Clegg23.update_caa("Na", "OH", "SO4", prm.psi_Na_OH_SO4_HMW84)
