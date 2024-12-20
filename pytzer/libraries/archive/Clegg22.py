# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2024  M.P. Humphreys  (GNU GPLv3)
from . import ParameterLibrary
from .. import debyehueckel, dissociation as k, parameters as prm, unsymmetrical

# First section is a direct copy of the Humphreys22 library
# ---------------------------------------------------------

# Following Supp. Info. part 6
Clegg22 = ParameterLibrary(name="Clegg22")
Clegg22.assign_func_J(unsymmetrical.P75_eq47)

# Table S6 (Aphi and equilibria)
Clegg22.update_Aphi(debyehueckel.Aosm_M88)
Clegg22.update_equilibrium("H2O", k.H2O_MF)
Clegg22.update_equilibrium("HSO4", k.HSO4_CRP94)
Clegg22.update_equilibrium("MgOH", k.MgOH_CW91_ln)

# Tables S7-S11 (beta and C coefficients)
Clegg22.update_ca("Ca", "Cl", prm.bC_Ca_Cl_GM89)
Clegg22.update_ca("Ca", "HSO4", prm.bC_Ca_HSO4_WM13)
Clegg22.update_ca("Ca", "OH", prm.bC_Ca_OH_HMW84)
Clegg22.update_ca("Ca", "SO4", prm.bC_Ca_SO4_WM13)
Clegg22.update_ca("H", "Cl", prm.bC_H_Cl_CMR93)
Clegg22.update_ca("H", "HSO4", prm.bC_H_HSO4_CRP94)
Clegg22.update_ca("H", "SO4", prm.bC_H_SO4_CRP94)
Clegg22.update_ca("K", "Cl", prm.bC_K_Cl_GM89)
Clegg22.update_ca("K", "HSO4", prm.bC_K_HSO4_WM13)
Clegg22.update_ca("K", "OH", prm.bC_K_OH_HMW84)
Clegg22.update_ca("K", "SO4", prm.bC_K_SO4_HM86)
Clegg22.update_ca("Mg", "Cl", prm.bC_Mg_Cl_dLP83)
Clegg22.update_ca("Mg", "HSO4", prm.bC_Mg_HSO4_RC99)
Clegg22.update_ca("Mg", "SO4", prm.bC_Mg_SO4_PP86ii)
Clegg22.update_ca("MgOH", "Cl", prm.bC_MgOH_Cl_HMW84)
Clegg22.update_ca("Na", "Cl", prm.bC_Na_Cl_M88)
Clegg22.update_ca("Na", "HSO4", prm.bC_Na_HSO4_HPR93)
Clegg22.update_ca("Na", "OH", prm.bC_Na_OH_HWT22)
Clegg22.update_ca("Na", "SO4", prm.bC_Na_SO4_HM86)

# Table S12 (cc theta and psi coefficients)
Clegg22.update_cc("Ca", "H", prm.theta_Ca_H_RGO81)
Clegg22.update_cca("Ca", "H", "Cl", prm.psi_Ca_H_Cl_HMW84)
Clegg22.update_cc("Ca", "K", prm.theta_Ca_K_HMW84)
Clegg22.update_cca("Ca", "K", "Cl", prm.psi_Ca_K_Cl_HMW84)
Clegg22.update_cc("Ca", "Mg", prm.theta_Ca_Mg_HMW84)
Clegg22.update_cca("Ca", "Mg", "Cl", prm.psi_Ca_Mg_Cl_HMW84)
Clegg22.update_cca("Ca", "Mg", "SO4", prm.psi_Ca_Mg_SO4_HMW84)
Clegg22.update_cc("Ca", "Na", prm.theta_Ca_Na_HMW84)
Clegg22.update_cca("Ca", "Na", "Cl", prm.psi_Ca_Na_Cl_HMW84)
Clegg22.update_cca("Ca", "Na", "SO4", prm.psi_Ca_Na_SO4_HMW84)
Clegg22.update_cc("H", "K", prm.theta_H_K_HWT22)
Clegg22.update_cca("H", "K", "Cl", prm.psi_H_K_Cl_HMW84)
Clegg22.update_cca("H", "K", "HSO4", prm.psi_H_K_HSO4_HMW84)
Clegg22.update_cca("H", "K", "SO4", prm.psi_H_K_SO4_HMW84)
Clegg22.update_cc("H", "Mg", prm.theta_H_Mg_RGB80)
Clegg22.update_cca("H", "Mg", "Cl", prm.psi_H_Mg_Cl_HMW84)
Clegg22.update_cca("H", "Mg", "HSO4", prm.psi_H_Mg_HSO4_RC99)
Clegg22.update_cca("H", "Mg", "SO4", prm.psi_H_Mg_SO4_RC99)
Clegg22.update_cc("H", "Na", prm.theta_H_Na_HWT22)
Clegg22.update_cca("H", "Na", "Cl", prm.psi_H_Na_Cl_HMW84)
Clegg22.update_cca("H", "Na", "HSO4", prm.psi_H_Na_HSO4_HMW84)
Clegg22.update_cc("K", "Mg", prm.theta_K_Mg_HMW84)
Clegg22.update_cca("K", "Mg", "Cl", prm.psi_K_Mg_Cl_HMW84)
Clegg22.update_cca("K", "Mg", "SO4", prm.psi_K_Mg_SO4_HMW84)
Clegg22.update_cc("K", "Na", prm.theta_K_Na_HMW84)
Clegg22.update_cca("K", "Na", "Cl", prm.psi_K_Na_Cl_HMW84)
Clegg22.update_cca("K", "Na", "SO4", prm.psi_K_Na_SO4_HMW84)
Clegg22.update_cc("Mg", "MgOH", prm.theta_Mg_MgOH_HMW84)
Clegg22.update_cca("Mg", "MgOH", "Cl", prm.psi_Mg_MgOH_Cl_HMW84)
Clegg22.update_cc("Mg", "Na", prm.theta_Mg_Na_HMW84)
Clegg22.update_cca("Mg", "Na", "Cl", prm.psi_Mg_Na_Cl_HMW84)
Clegg22.update_cca("Mg", "Na", "SO4", prm.psi_Mg_Na_SO4_HMW84)

# Table S13 (aa theta and psi coefficients)
Clegg22.update_aa("Cl", "HSO4", prm.theta_Cl_HSO4_HMW84)
Clegg22.update_caa("Na", "Cl", "HSO4", prm.psi_Na_Cl_HSO4_HMW84)
Clegg22.update_caa("H", "Cl", "HSO4", prm.psi_H_Cl_HSO4_HMW84)
Clegg22.update_aa("Cl", "OH", prm.theta_Cl_OH_HMW84)
Clegg22.update_caa("Ca", "Cl", "OH", prm.psi_Ca_Cl_OH_HMW84)
Clegg22.update_caa("K", "Cl", "OH", prm.psi_K_Cl_OH_HMW84)
Clegg22.update_caa("Na", "Cl", "OH", prm.psi_Na_Cl_OH_HMW84)
Clegg22.update_aa("Cl", "SO4", prm.theta_Cl_SO4_HMW84)
Clegg22.update_caa("Ca", "Cl", "SO4", prm.psi_Ca_Cl_SO4_HMW84)
Clegg22.update_caa("Mg", "Cl", "SO4", prm.psi_Mg_Cl_SO4_HMW84)
Clegg22.update_caa("Na", "Cl", "SO4", prm.psi_Na_Cl_SO4_HMW84)
Clegg22.update_aa("HSO4", "SO4", prm.theta_HSO4_SO4_WM13)
Clegg22.update_caa("K", "HSO4", "SO4", prm.psi_K_HSO4_SO4_HMW84)
Clegg22.update_caa("Mg", "HSO4", "SO4", prm.psi_Mg_HSO4_SO4_RC99)
Clegg22.update_caa("Na", "HSO4", "SO4", prm.psi_Na_HSO4_SO4_HMW84)
Clegg22.update_aa("OH", "SO4", prm.theta_OH_SO4_HMW84)
Clegg22.update_caa("K", "OH", "SO4", prm.psi_K_OH_SO4_HMW84)
Clegg22.update_caa("Na", "OH", "SO4", prm.psi_Na_OH_SO4_HMW84)

# Below here is new things from the CHW22 Supp. Info.
# ---------------------------------------------------

# Table S7 (equilibrium constants)
Clegg22.update_equilibrium("trisH", k.trisH_BH61)

# Tables S8-S12 (betas and Cs)
Clegg22.update_ca("trisH", "Cl", prm.bC_trisH_Cl_CHW22)
Clegg22.update_ca("trisH", "SO4", prm.bC_trisH_SO4_CHW22)

# Table S15 (lambdas and mu)
Clegg22.update_nc("tris", "Ca", prm.lambd_tris_Ca_CHW22)
Clegg22.update_nc("tris", "K", prm.lambd_tris_K_CHW22)
Clegg22.update_nc("tris", "Mg", prm.lambd_tris_Mg_CHW22)
Clegg22.update_nc("tris", "Na", prm.lambd_tris_Na_CHW22)
Clegg22.update_na("tris", "trisH", prm.lambd_tris_trisH_LTA21)
Clegg22.update_na("tris", "SO4", prm.lambd_tris_SO4_LTA21)
Clegg22.update_nn("tris", "tris", prm.lambd_tris_tris_LTA21)
Clegg22.update_nnn("tris", prm.mu_tris_tris_tris_LTA21)
