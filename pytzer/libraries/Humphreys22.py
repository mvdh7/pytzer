# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2023  Matthew P. Humphreys  (GNU GPLv3)
from . import ParameterLibrary
from .. import debyehueckel, dissociation as k, parameters as prm, unsymmetrical

# Following Supp. Info. part 6
Humphreys22 = ParameterLibrary(name="Humphreys22")
Humphreys22.assign_func_J(unsymmetrical.P75_eq47)

# Table S6 (Aphi and equilibria)
Humphreys22.update_Aphi(debyehueckel.Aosm_M88)
Humphreys22.update_equilibrium("H2O", k.H2O_MF)
Humphreys22.update_equilibrium("HSO4", k.HSO4_CRP94)
Humphreys22.update_equilibrium("MgOH", k.MgOH_CW91_ln)

# Tables S7-S11 (beta and C coefficients)
Humphreys22.update_ca("Ca", "Cl", prm.bC_Ca_Cl_GM89)
Humphreys22.update_ca("Ca", "HSO4", prm.bC_Ca_HSO4_WM13)
Humphreys22.update_ca("Ca", "OH", prm.bC_Ca_OH_HMW84)
Humphreys22.update_ca("Ca", "SO4", prm.bC_Ca_SO4_WM13)
Humphreys22.update_ca("H", "Cl", prm.bC_H_Cl_CMR93)
Humphreys22.update_ca("H", "HSO4", prm.bC_H_HSO4_CRP94)
Humphreys22.update_ca("H", "SO4", prm.bC_H_SO4_CRP94)
Humphreys22.update_ca("K", "Cl", prm.bC_K_Cl_GM89)
Humphreys22.update_ca("K", "HSO4", prm.bC_K_HSO4_WM13)
Humphreys22.update_ca("K", "OH", prm.bC_K_OH_HMW84)
Humphreys22.update_ca("K", "SO4", prm.bC_K_SO4_HM86)
Humphreys22.update_ca("Mg", "Cl", prm.bC_Mg_Cl_dLP83)
Humphreys22.update_ca("Mg", "HSO4", prm.bC_Mg_HSO4_RC99)
Humphreys22.update_ca("Mg", "SO4", prm.bC_Mg_SO4_PP86ii)
Humphreys22.update_ca("MgOH", "Cl", prm.bC_MgOH_Cl_HMW84)
Humphreys22.update_ca("Na", "Cl", prm.bC_Na_Cl_M88)
Humphreys22.update_ca("Na", "HSO4", prm.bC_Na_HSO4_HPR93)
Humphreys22.update_ca("Na", "OH", prm.bC_Na_OH_HWT22)
Humphreys22.update_ca("Na", "SO4", prm.bC_Na_SO4_HM86)

# Table S12 (cc theta and psi coefficients)
Humphreys22.update_cc("Ca", "H", prm.theta_Ca_H_RGO81)
Humphreys22.update_cca("Ca", "H", "Cl", prm.psi_Ca_H_Cl_HMW84)
Humphreys22.update_cc("Ca", "K", prm.theta_Ca_K_HMW84)
Humphreys22.update_cca("Ca", "K", "Cl", prm.psi_Ca_K_Cl_HMW84)
Humphreys22.update_cc("Ca", "Mg", prm.theta_Ca_Mg_HMW84)
Humphreys22.update_cca("Ca", "Mg", "Cl", prm.psi_Ca_Mg_Cl_HMW84)
Humphreys22.update_cca("Ca", "Mg", "SO4", prm.psi_Ca_Mg_SO4_HMW84)
Humphreys22.update_cc("Ca", "Na", prm.theta_Ca_Na_HMW84)
Humphreys22.update_cca("Ca", "Na", "Cl", prm.psi_Ca_Na_Cl_HMW84)
Humphreys22.update_cca("Ca", "Na", "SO4", prm.psi_Ca_Na_SO4_HMW84)
Humphreys22.update_cc("H", "K", prm.theta_H_K_HWT22)
Humphreys22.update_cca("H", "K", "Cl", prm.psi_H_K_Cl_HMW84)
Humphreys22.update_cca("H", "K", "HSO4", prm.psi_H_K_HSO4_HMW84)
Humphreys22.update_cca("H", "K", "SO4", prm.psi_H_K_SO4_HMW84)
Humphreys22.update_cc("H", "Mg", prm.theta_H_Mg_RGB80)
Humphreys22.update_cca("H", "Mg", "Cl", prm.psi_H_Mg_Cl_HMW84)
Humphreys22.update_cca("H", "Mg", "HSO4", prm.psi_H_Mg_HSO4_RC99)
Humphreys22.update_cca("H", "Mg", "SO4", prm.psi_H_Mg_SO4_RC99)
Humphreys22.update_cc("H", "Na", prm.theta_H_Na_HWT22)
Humphreys22.update_cca("H", "Na", "Cl", prm.psi_H_Na_Cl_HMW84)
Humphreys22.update_cca("H", "Na", "HSO4", prm.psi_H_Na_HSO4_HMW84)
Humphreys22.update_cc("K", "Mg", prm.theta_K_Mg_HMW84)
Humphreys22.update_cca("K", "Mg", "Cl", prm.psi_K_Mg_Cl_HMW84)
Humphreys22.update_cca("K", "Mg", "SO4", prm.psi_K_Mg_SO4_HMW84)
Humphreys22.update_cc("K", "Na", prm.theta_K_Na_HMW84)
Humphreys22.update_cca("K", "Na", "Cl", prm.psi_K_Na_Cl_HMW84)
Humphreys22.update_cca("K", "Na", "SO4", prm.psi_K_Na_SO4_HMW84)
Humphreys22.update_cc("Mg", "MgOH", prm.theta_Mg_MgOH_HMW84)
Humphreys22.update_cca("Mg", "MgOH", "Cl", prm.psi_Mg_MgOH_Cl_HMW84)
Humphreys22.update_cc("Mg", "Na", prm.theta_Mg_Na_HMW84)
Humphreys22.update_cca("Mg", "Na", "Cl", prm.psi_Mg_Na_Cl_HMW84)
Humphreys22.update_cca("Mg", "Na", "SO4", prm.psi_Mg_Na_SO4_HMW84)

# Table S13 (aa theta and psi coefficients)
Humphreys22.update_aa("Cl", "HSO4", prm.theta_Cl_HSO4_HMW84)
Humphreys22.update_caa("Na", "Cl", "HSO4", prm.psi_Na_Cl_HSO4_HMW84)
Humphreys22.update_caa("H", "Cl", "HSO4", prm.psi_H_Cl_HSO4_HMW84)
Humphreys22.update_aa("Cl", "OH", prm.theta_Cl_OH_HMW84)
Humphreys22.update_caa("Ca", "Cl", "OH", prm.psi_Ca_Cl_OH_HMW84)
Humphreys22.update_caa("K", "Cl", "OH", prm.psi_K_Cl_OH_HMW84)
Humphreys22.update_caa("Na", "Cl", "OH", prm.psi_Na_Cl_OH_HMW84)
Humphreys22.update_aa("Cl", "SO4", prm.theta_Cl_SO4_HMW84)
Humphreys22.update_caa("Ca", "Cl", "SO4", prm.psi_Ca_Cl_SO4_HMW84)
Humphreys22.update_caa("Mg", "Cl", "SO4", prm.psi_Mg_Cl_SO4_HMW84)
Humphreys22.update_caa("Na", "Cl", "SO4", prm.psi_Na_Cl_SO4_HMW84)
Humphreys22.update_aa("HSO4", "SO4", prm.theta_HSO4_SO4_WM13)
Humphreys22.update_caa("K", "HSO4", "SO4", prm.psi_K_HSO4_SO4_HMW84)
Humphreys22.update_caa("Mg", "HSO4", "SO4", prm.psi_Mg_HSO4_SO4_RC99)
Humphreys22.update_caa("Na", "HSO4", "SO4", prm.psi_Na_HSO4_SO4_HMW84)
Humphreys22.update_aa("OH", "SO4", prm.theta_OH_SO4_HMW84)
Humphreys22.update_caa("K", "OH", "SO4", prm.psi_K_OH_SO4_HMW84)
Humphreys22.update_caa("Na", "OH", "SO4", prm.psi_Na_OH_SO4_HMW84)
