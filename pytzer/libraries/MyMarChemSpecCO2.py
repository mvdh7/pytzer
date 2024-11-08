# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2024  Matthew P. Humphreys  (GNU GPLv3)
"""Waters & Millero (2013).
*Marine Chemistry* 149, 8--22.
doi:10.1016/j.marchem.2012.11.003
according to Clegg et al. (2022)
with CO2-system terms added
"""
from . import ParameterLibrary
from .. import debyehueckel, dissociation as k, parameters as prm, unsymmetrical

MyMarChemSpecCO2 = ParameterLibrary(name="MyMarChemSpecCO2")
MyMarChemSpecCO2.update_Aphi(debyehueckel.Aosm_M88)
MyMarChemSpecCO2.assign_func_J(unsymmetrical.Harvie)
# Table A1: Na salts
MyMarChemSpecCO2.update_ca("Na", "Cl", prm.bC_Na_Cl_M88)
# MyMarChemSpecCO2.update_ca("Na", "SO4", prm.bC_Na_SO4_HM86)
# MyMarChemSpecCO2.update_ca("Na", "HSO4", prm.bC_Na_HSO4_HPR93)
MyMarChemSpecCO2.update_ca("Na", "OH", prm.bC_Na_OH_PP87i)
# # Table A2: Mg salts
# MyMarChemSpecCO2.update_ca("Mg", "Cl", prm.bC_Mg_Cl_dLP83)
# MyMarChemSpecCO2.update_ca("Mg", "SO4", prm.bC_Mg_SO4_PP86ii)
# MyMarChemSpecCO2.update_ca("Mg", "HSO4", prm.bC_Mg_HSO4_RC99)
# # Table A3: Ca salts
# MyMarChemSpecCO2.update_ca("Ca", "Cl", prm.bC_Ca_Cl_GM89)
# MyMarChemSpecCO2.update_ca("Ca", "SO4", prm.bC_Ca_SO4_WM13)
# MyMarChemSpecCO2.update_ca("Ca", "HSO4", prm.bC_Ca_HSO4_WM13)
# MyMarChemSpecCO2.update_ca("Ca", "OH", prm.bC_Ca_OH_HMW84)
# # Table A4: K salts
# MyMarChemSpecCO2.update_ca("K", "Cl", prm.bC_K_Cl_GM89)
# MyMarChemSpecCO2.update_ca("K", "SO4", prm.bC_K_SO4_HM86)
# MyMarChemSpecCO2.update_ca("K", "HSO4", prm.bC_K_HSO4_WM13)
# MyMarChemSpecCO2.update_ca("K", "OH", prm.bC_K_OH_HMW84)
# Table A5: H+ interactions
MyMarChemSpecCO2.update_ca("H", "Cl", prm.bC_H_Cl_CMR93)
# MyMarChemSpecCO2.update_ca("H", "SO4", prm.bC_H_SO4_CRP94)
# MyMarChemSpecCO2.update_ca("H", "HSO4", prm.bC_H_HSO4_CRP94)
# # Table A6: MgOH+ interactions
# MyMarChemSpecCO2.update_ca("MgOH", "Cl", prm.bC_MgOH_Cl_HMW84)
# Table A7: cation-cation interactions
MyMarChemSpecCO2.update_cc("H", "Na", prm.theta_H_Na_HCW22)  # different from WM13
# MyMarChemSpecCO2.update_cc("H", "Mg", prm.theta_H_Mg_RGB80)
# MyMarChemSpecCO2.update_cc("Ca", "H", prm.theta_Ca_H_RGO81)
# MyMarChemSpecCO2.update_cc("H", "K", prm.theta_H_K_CMR93)
# MyMarChemSpecCO2.update_cc("Mg", "Na", prm.theta_Mg_Na_HMW84)
# MyMarChemSpecCO2.update_cc("Ca", "Na", prm.theta_Ca_Na_HMW84)
# MyMarChemSpecCO2.update_cc("K", "Na", prm.theta_K_Na_HMW84)
# MyMarChemSpecCO2.update_cc("Ca", "Mg", prm.theta_Ca_Mg_HMW84)
# MyMarChemSpecCO2.update_cc("K", "Mg", prm.theta_K_Mg_HMW84)
# MyMarChemSpecCO2.update_cc("Ca", "K", prm.theta_Ca_K_HMW84)
# # Table A7: anion-anion interactions
# MyMarChemSpecCO2.update_aa("Cl", "SO4", prm.theta_Cl_SO4_HMW84)
# MyMarChemSpecCO2.update_aa("Cl", "HSO4", prm.theta_Cl_HSO4_HMW84)
MyMarChemSpecCO2.update_aa("Cl", "OH", prm.theta_Cl_OH_HMW84)
# MyMarChemSpecCO2.update_aa("HSO4", "SO4", prm.theta_HSO4_SO4_WM13)
# MyMarChemSpecCO2.update_aa("OH", "SO4", prm.theta_OH_SO4_HMW84)
# Table A8: c-a-a' triplets
# MyMarChemSpecCO2.update_caa("H", "Cl", "SO4", prm.psi_H_Cl_SO4_WM13)
# MyMarChemSpecCO2.update_caa("Na", "Cl", "SO4", prm.psi_Na_Cl_SO4_HMW84)
# MyMarChemSpecCO2.update_caa("Mg", "Cl", "SO4", prm.psi_Mg_Cl_SO4_HMW84)
# MyMarChemSpecCO2.update_caa("Ca", "Cl", "SO4", prm.psi_Ca_Cl_SO4_HMW84)
# MyMarChemSpecCO2.update_caa("K", "Cl", "SO4", prm.psi_K_Cl_SO4_HMW84)
# MyMarChemSpecCO2.update_caa("H", "Cl", "HSO4", prm.psi_H_Cl_HSO4_HMW84)
# MyMarChemSpecCO2.update_caa("Na", "Cl", "HSO4", prm.psi_Na_Cl_HSO4_HMW84)
# MyMarChemSpecCO2.update_caa("Mg", "Cl", "HSO4", prm.psi_Mg_Cl_HSO4_HMW84)
# MyMarChemSpecCO2.update_caa("Ca", "Cl", "HSO4", prm.psi_Ca_Cl_HSO4_HMW84)
# MyMarChemSpecCO2.update_caa("K", "Cl", "HSO4", prm.psi_K_Cl_HSO4_HMW84)
MyMarChemSpecCO2.update_caa("H", "Cl", "OH", prm.psi_H_Cl_OH_WM13)
MyMarChemSpecCO2.update_caa("Na", "Cl", "OH", prm.psi_Na_Cl_OH_HMW84)
# MyMarChemSpecCO2.update_caa("Mg", "Cl", "OH", prm.psi_Mg_Cl_OH_WM13)
# MyMarChemSpecCO2.update_caa("Ca", "Cl", "OH", prm.psi_Ca_Cl_OH_HMW84)
# MyMarChemSpecCO2.update_caa("K", "Cl", "OH", prm.psi_K_Cl_OH_HMW84)
# MyMarChemSpecCO2.update_caa("H", "HSO4", "SO4", prm.psi_H_HSO4_SO4_HMW84)
# MyMarChemSpecCO2.update_caa("Na", "HSO4", "SO4", prm.psi_Na_HSO4_SO4_HMW84)
# MyMarChemSpecCO2.update_caa("Mg", "HSO4", "SO4", prm.psi_Mg_HSO4_SO4_RC99)
# MyMarChemSpecCO2.update_caa("Ca", "HSO4", "SO4", prm.psi_Ca_HSO4_SO4_WM13)
# MyMarChemSpecCO2.update_caa("K", "HSO4", "SO4", prm.psi_K_HSO4_SO4_HMW84)
# MyMarChemSpecCO2.update_caa("H", "OH", "SO4", prm.psi_H_OH_SO4_WM13)
# MyMarChemSpecCO2.update_caa("Na", "OH", "SO4", prm.psi_Na_OH_SO4_HMW84)
# MyMarChemSpecCO2.update_caa("Mg", "OH", "SO4", prm.psi_Mg_OH_SO4_WM13)
# MyMarChemSpecCO2.update_caa("Ca", "OH", "SO4", prm.psi_Ca_OH_SO4_WM13)
# MyMarChemSpecCO2.update_caa("K", "OH", "SO4", prm.psi_K_OH_SO4_HMW84)
# Table A9: c-c'-a triplets
MyMarChemSpecCO2.update_cca("H", "Na", "Cl", prm.psi_H_Na_Cl_HMW84)
# MyMarChemSpecCO2.update_cca("H", "Na", "SO4", prm.psi_H_Na_SO4_WM13)
# MyMarChemSpecCO2.update_cca("H", "Na", "HSO4", prm.psi_H_Na_HSO4_HMW84)
# MyMarChemSpecCO2.update_cca("H", "Mg", "Cl", prm.psi_H_Mg_Cl_HMW84)
# MyMarChemSpecCO2.update_cca("H", "Mg", "SO4", prm.psi_H_Mg_SO4_RC99)
# MyMarChemSpecCO2.update_cca("H", "Mg", "HSO4", prm.psi_H_Mg_HSO4_RC99)
# MyMarChemSpecCO2.update_cca("Ca", "H", "Cl", prm.psi_Ca_H_Cl_HMW84)
# MyMarChemSpecCO2.update_cca("Ca", "H", "SO4", prm.psi_Ca_H_SO4_WM13)
# MyMarChemSpecCO2.update_cca("Ca", "H", "HSO4", prm.psi_Ca_H_HSO4_WM13)
# MyMarChemSpecCO2.update_cca("H", "K", "Cl", prm.psi_H_K_Cl_HMW84)
# MyMarChemSpecCO2.update_cca("H", "K", "SO4", prm.psi_H_K_SO4_HMW84)
# MyMarChemSpecCO2.update_cca("H", "K", "HSO4", prm.psi_H_K_HSO4_HMW84)
# MyMarChemSpecCO2.update_cca("Mg", "Na", "Cl", prm.psi_Mg_Na_Cl_HMW84)
# MyMarChemSpecCO2.update_cca("Mg", "Na", "SO4", prm.psi_Mg_Na_SO4_HMW84)
# MyMarChemSpecCO2.update_cca("Mg", "Na", "HSO4", prm.psi_Mg_Na_HSO4_WM13)
# MyMarChemSpecCO2.update_cca("Ca", "Na", "Cl", prm.psi_Ca_Na_Cl_HMW84)
# MyMarChemSpecCO2.update_cca("Ca", "Na", "SO4", prm.psi_Ca_Na_SO4_HMW84)
# MyMarChemSpecCO2.update_cca("Ca", "Na", "HSO4", prm.psi_Ca_Na_HSO4_WM13)
# MyMarChemSpecCO2.update_cca("K", "Na", "Cl", prm.psi_K_Na_Cl_HMW84)
# MyMarChemSpecCO2.update_cca("K", "Na", "SO4", prm.psi_K_Na_SO4_HMW84)
# MyMarChemSpecCO2.update_cca("K", "Na", "HSO4", prm.psi_K_Na_HSO4_WM13)
# MyMarChemSpecCO2.update_cca("Ca", "Mg", "Cl", prm.psi_Ca_Mg_Cl_HMW84)
# MyMarChemSpecCO2.update_cca("Ca", "Mg", "SO4", prm.psi_Ca_Mg_SO4_HMW84)
# MyMarChemSpecCO2.update_cca("Ca", "Mg", "HSO4", prm.psi_Ca_Mg_HSO4_WM13)
# MyMarChemSpecCO2.update_cca("K", "Mg", "Cl", prm.psi_K_Mg_Cl_HMW84)
# MyMarChemSpecCO2.update_cca("K", "Mg", "SO4", prm.psi_K_Mg_SO4_HMW84)
# MyMarChemSpecCO2.update_cca("K", "Mg", "HSO4", prm.psi_K_Mg_HSO4_WM13)
# MyMarChemSpecCO2.update_cca("Ca", "K", "Cl", prm.psi_Ca_K_Cl_HMW84)
# MyMarChemSpecCO2.update_cca("Ca", "K", "SO4", prm.psi_Ca_K_SO4_WM13)
# MyMarChemSpecCO2.update_cca("Ca", "K", "HSO4", prm.psi_Ca_K_HSO4_WM13)
# MyMarChemSpecCO2.update_cca("Mg", "MgOH", "Cl", prm.psi_Mg_MgOH_Cl_HMW84)  # not in WM13
# Add equilibria
# MyMarChemSpecCO2.update_equilibrium("HSO4", k.HSO4_CRP94)
# MyMarChemSpecCO2.update_equilibrium("MgOH", k.MgOH_CW91_ln)
MyMarChemSpecCO2.update_equilibrium("H2O", k.H2O_M79)
# # Extras from Clegg22 (all before this point is from Humphreys22)
# MyMarChemSpecCO2.update_equilibrium("trisH", k.trisH_BH61)
# MyMarChemSpecCO2.update_ca("trisH", "Cl", prm.bC_trisH_Cl_CHW22)
# MyMarChemSpecCO2.update_ca("trisH", "SO4", prm.bC_trisH_SO4_CHW22)
# MyMarChemSpecCO2.update_nc("tris", "Ca", prm.lambd_tris_Ca_CHW22)
# MyMarChemSpecCO2.update_nc("tris", "K", prm.lambd_tris_K_CHW22)
# MyMarChemSpecCO2.update_nc("tris", "Mg", prm.lambd_tris_Mg_CHW22)
# MyMarChemSpecCO2.update_nc("tris", "Na", prm.lambd_tris_Na_CHW22)
# MyMarChemSpecCO2.update_nc("tris", "trisH", prm.lambd_tris_trisH_LTA21)
# MyMarChemSpecCO2.update_na("tris", "SO4", prm.lambd_tris_SO4_LTA21)
# MyMarChemSpecCO2.update_nn("tris", "tris", prm.lambd_tris_tris_LTA21)
# MyMarChemSpecCO2.update_nnn("tris", prm.mu_tris_tris_tris_LTA21)

MyMarChemSpecCO2.update_equilibrium("H2CO3", k.H2CO3_PB82)
MyMarChemSpecCO2.update_equilibrium("HCO3", k.HCO3_PB82)
# still needs CaCO3, MgCO3 formation!

# Added CO2 system from HM93
MyMarChemSpecCO2.update_ca("Na", "HCO3", prm.bC_Na_HCO3_HM93)
MyMarChemSpecCO2.update_ca("Na", "CO3", prm.bC_Na_CO3_HM93)
# MyMarChemSpecCO2.update_ca("K", "HCO3", prm.bC_K_HCO3_HM93)
# MyMarChemSpecCO2.update_ca("K", "CO3", prm.bC_K_CO3_HM93)
# MyMarChemSpecCO2.update_ca("Ca", "HCO3", prm.bC_Ca_HCO3_HM93)
# MyMarChemSpecCO2.update_ca("Ca", "CO3", prm.bC_Ca_CO3_HM93)
# MyMarChemSpecCO2.update_ca("Mg", "HCO3", prm.bC_Mg_HCO3_HM93)
# MyMarChemSpecCO2.update_ca("Mg", "CO3", prm.bC_Mg_CO3_HM93)
# MyMarChemSpecCO2.update_ca("MgOH", "HCO3", prm.bC_MgOH_HCO3_HMW84)
# MyMarChemSpecCO2.update_ca("MgOH", "CO3", prm.bC_MgOH_CO3_HMW84)
MyMarChemSpecCO2.update_ca("H", "HCO3", prm.bC_H_HCO3_HMW84)
MyMarChemSpecCO2.update_ca("H", "CO3", prm.bC_H_CO3_HMW84)
# thetas and psis
MyMarChemSpecCO2.update_aa("Cl", "HCO3", prm.theta_Cl_HCO3_HMW84)
MyMarChemSpecCO2.update_caa("Na", "Cl", "HCO3", prm.psi_Na_Cl_HCO3_HMW84)
# MyMarChemSpecCO2.update_caa("K", "Cl", "HCO3", prm.psi_K_Cl_HCO3_HMW84)
# MyMarChemSpecCO2.update_caa("Ca", "Cl", "HCO3", prm.psi_Ca_Cl_HCO3_HMW84)
# MyMarChemSpecCO2.update_caa("Mg", "Cl", "HCO3", prm.psi_Mg_Cl_HCO3_HMW84)
# MyMarChemSpecCO2.update_caa("MgOH", "Cl", "HCO3", prm.psi_MgOH_Cl_HCO3_HMW84)
MyMarChemSpecCO2.update_caa("H", "Cl", "HCO3", prm.psi_H_Cl_HCO3_HMW84)
MyMarChemSpecCO2.update_aa("CO3", "Cl", prm.theta_CO3_Cl_HMW84)
MyMarChemSpecCO2.update_caa("Na", "CO3", "Cl", prm.psi_Na_CO3_Cl_HMW84)
# MyMarChemSpecCO2.update_caa("K", "CO3", "Cl", prm.psi_K_CO3_Cl_HMW84)
# MyMarChemSpecCO2.update_caa("Ca", "CO3", "Cl", prm.psi_Ca_CO3_Cl_HMW84)
# MyMarChemSpecCO2.update_caa("Mg", "CO3", "Cl", prm.psi_Mg_CO3_Cl_HMW84)
# MyMarChemSpecCO2.update_caa("MgOH", "CO3", "Cl", prm.psi_MgOH_CO3_Cl_HMW84)
MyMarChemSpecCO2.update_caa("H", "CO3", "Cl", prm.psi_H_CO3_Cl_HMW84)
# MyMarChemSpecCO2.update_aa("HCO3", "SO4", prm.theta_HCO3_SO4_HMW84)
# MyMarChemSpecCO2.update_caa("Na", "HCO3", "SO4", prm.psi_Na_HCO3_SO4_HMW84)
# MyMarChemSpecCO2.update_caa("K", "HCO3", "SO4", prm.psi_K_HCO3_SO4_HMW84)
# MyMarChemSpecCO2.update_caa("Ca", "HCO3", "SO4", prm.psi_Ca_HCO3_SO4_HMW84)
# MyMarChemSpecCO2.update_caa("Mg", "HCO3", "SO4", prm.psi_Mg_HCO3_SO4_HMW84)
# MyMarChemSpecCO2.update_caa("MgOH", "HCO3", "SO4", prm.psi_MgOH_HCO3_SO4_HMW84)
# MyMarChemSpecCO2.update_caa("H", "HCO3", "SO4", prm.psi_H_HCO3_SO4_HMW84)
# MyMarChemSpecCO2.update_aa("CO3", "SO4", prm.theta_CO3_SO4_HMW84)
# MyMarChemSpecCO2.update_caa("Na", "CO3", "SO4", prm.psi_Na_CO3_SO4_HMW84)
# MyMarChemSpecCO2.update_caa("K", "CO3", "SO4", prm.psi_K_CO3_SO4_HMW84)
# MyMarChemSpecCO2.update_caa("Ca", "CO3", "SO4", prm.psi_Ca_CO3_SO4_HMW84)
# MyMarChemSpecCO2.update_caa("Mg", "CO3", "SO4", prm.psi_Mg_CO3_SO4_HMW84)
# MyMarChemSpecCO2.update_caa("MgOH", "CO3", "SO4", prm.psi_MgOH_CO3_SO4_HMW84)
# MyMarChemSpecCO2.update_caa("H", "CO3", "SO4", prm.psi_H_CO3_SO4_HMW84)
# MyMarChemSpecCO2.update_aa("HCO3", "HSO4", prm.theta_HCO3_HSO4_HMW84)
# MyMarChemSpecCO2.update_caa("Na", "HCO3", "HSO4", prm.psi_Na_HCO3_HSO4_HMW84)
# MyMarChemSpecCO2.update_caa("K", "HCO3", "HSO4", prm.psi_K_HCO3_HSO4_HMW84)
# MyMarChemSpecCO2.update_caa("Ca", "HCO3", "HSO4", prm.psi_Ca_HCO3_HSO4_HMW84)
# MyMarChemSpecCO2.update_caa("Mg", "HCO3", "HSO4", prm.psi_Mg_HCO3_HSO4_HMW84)
# MyMarChemSpecCO2.update_caa("MgOH", "HCO3", "HSO4", prm.psi_MgOH_HCO3_HSO4_HMW84)
# MyMarChemSpecCO2.update_caa("H", "HCO3", "HSO4", prm.psi_H_HCO3_HSO4_HMW84)
# MyMarChemSpecCO2.update_aa("CO3", "HSO4", prm.theta_CO3_HSO4_HMW84)
# MyMarChemSpecCO2.update_caa("Na", "CO3", "HSO4", prm.psi_Na_CO3_HSO4_HMW84)
# MyMarChemSpecCO2.update_caa("K", "CO3", "HSO4", prm.psi_K_CO3_HSO4_HMW84)
# MyMarChemSpecCO2.update_caa("Ca", "CO3", "HSO4", prm.psi_Ca_CO3_HSO4_HMW84)
# MyMarChemSpecCO2.update_caa("Mg", "CO3", "HSO4", prm.psi_Mg_CO3_HSO4_HMW84)
# MyMarChemSpecCO2.update_caa("MgOH", "CO3", "HSO4", prm.psi_MgOH_CO3_HSO4_HMW84)
# MyMarChemSpecCO2.update_caa("H", "CO3", "HSO4", prm.psi_H_CO3_HSO4_HMW84)
MyMarChemSpecCO2.update_aa("HCO3", "OH", prm.theta_HCO3_OH_HMW84)
MyMarChemSpecCO2.update_caa("Na", "HCO3", "OH", prm.psi_Na_HCO3_OH_HMW84)
# MyMarChemSpecCO2.update_caa("K", "HCO3", "OH", prm.psi_K_HCO3_OH_HMW84)
# MyMarChemSpecCO2.update_caa("Ca", "HCO3", "OH", prm.psi_Ca_HCO3_OH_HMW84)
# MyMarChemSpecCO2.update_caa("Mg", "HCO3", "OH", prm.psi_Mg_HCO3_OH_HMW84)
# MyMarChemSpecCO2.update_caa("MgOH", "HCO3", "OH", prm.psi_MgOH_HCO3_OH_HMW84)
MyMarChemSpecCO2.update_caa("H", "HCO3", "OH", prm.psi_H_HCO3_OH_HMW84)
MyMarChemSpecCO2.update_aa("CO3", "OH", prm.theta_CO3_OH_HMW84)
MyMarChemSpecCO2.update_caa("Na", "CO3", "OH", prm.psi_Na_CO3_OH_HMW84)
# MyMarChemSpecCO2.update_caa("K", "CO3", "OH", prm.psi_K_CO3_OH_HMW84)
# MyMarChemSpecCO2.update_caa("Ca", "CO3", "OH", prm.psi_Ca_CO3_OH_HMW84)
# MyMarChemSpecCO2.update_caa("Mg", "CO3", "OH", prm.psi_Mg_CO3_OH_HMW84)
# MyMarChemSpecCO2.update_caa("MgOH", "CO3", "OH", prm.psi_MgOH_CO3_OH_HMW84)
MyMarChemSpecCO2.update_caa("H", "CO3", "OH", prm.psi_H_CO3_OH_HMW84)
MyMarChemSpecCO2.update_aa("CO3", "HCO3", prm.theta_CO3_HCO3_HMW84)
MyMarChemSpecCO2.update_caa("Na", "CO3", "HCO3", prm.psi_Na_CO3_HCO3_HMW84)
# MyMarChemSpecCO2.update_caa("K", "CO3", "HCO3", prm.psi_K_CO3_HCO3_HMW84)
# MyMarChemSpecCO2.update_caa("Ca", "CO3", "HCO3", prm.psi_Ca_CO3_HCO3_HMW84)
# MyMarChemSpecCO2.update_caa("Mg", "CO3", "HCO3", prm.psi_Mg_CO3_HCO3_HMW84)
# MyMarChemSpecCO2.update_caa("MgOH", "CO3", "HCO3", prm.psi_MgOH_CO3_HCO3_HMW84)
MyMarChemSpecCO2.update_caa("H", "CO3", "HCO3", prm.psi_H_CO3_HCO3_HMW84)
# MyMarChemSpecCO2.update_cca("K", "Na", "HCO3", prm.psi_K_Na_HCO3_HMW84)
# MyMarChemSpecCO2.update_cca("K", "Na", "CO3", prm.psi_K_Na_CO3_HMW84)
# MyMarChemSpecCO2.update_cca("Ca", "Na", "HCO3", prm.psi_Ca_Na_HCO3_HMW84)
# MyMarChemSpecCO2.update_cca("Ca", "Na", "CO3", prm.psi_Ca_Na_CO3_HMW84)
# MyMarChemSpecCO2.update_cca("Mg", "Na", "HCO3", prm.psi_Mg_Na_HCO3_HMW84)
# MyMarChemSpecCO2.update_cca("Mg", "Na", "CO3", prm.psi_Mg_Na_CO3_HMW84)
# MyMarChemSpecCO2.update_cca("MgOH", "Na", "HCO3", prm.psi_MgOH_Na_HCO3_HMW84)
# MyMarChemSpecCO2.update_cca("MgOH", "Na", "CO3", prm.psi_MgOH_Na_CO3_HMW84)
MyMarChemSpecCO2.update_cca("H", "Na", "HCO3", prm.psi_H_Na_HCO3_HMW84)
MyMarChemSpecCO2.update_cca("H", "Na", "CO3", prm.psi_H_Na_CO3_HMW84)
# MyMarChemSpecCO2.update_cca("Ca", "K", "HCO3", prm.psi_Ca_K_HCO3_HMW84)
# MyMarChemSpecCO2.update_cca("Ca", "K", "CO3", prm.psi_Ca_K_CO3_HMW84)
# MyMarChemSpecCO2.update_cca("K", "Mg", "HCO3", prm.psi_K_Mg_HCO3_HMW84)
# MyMarChemSpecCO2.update_cca("K", "Mg", "CO3", prm.psi_K_Mg_CO3_HMW84)
# MyMarChemSpecCO2.update_cca("K", "MgOH", "HCO3", prm.psi_K_MgOH_HCO3_HMW84)
# MyMarChemSpecCO2.update_cca("K", "MgOH", "CO3", prm.psi_K_MgOH_CO3_HMW84)
# MyMarChemSpecCO2.update_cca("H", "K", "HCO3", prm.psi_H_K_HCO3_HMW84)
# MyMarChemSpecCO2.update_cca("H", "K", "CO3", prm.psi_H_K_CO3_HMW84)
# MyMarChemSpecCO2.update_cca("Ca", "Mg", "HCO3", prm.psi_Ca_Mg_HCO3_HMW84)
# MyMarChemSpecCO2.update_cca("Ca", "Mg", "CO3", prm.psi_Ca_Mg_CO3_HMW84)
# MyMarChemSpecCO2.update_cca("Ca", "MgOH", "HCO3", prm.psi_Ca_MgOH_HCO3_HMW84)
# MyMarChemSpecCO2.update_cca("Ca", "MgOH", "CO3", prm.psi_Ca_MgOH_CO3_HMW84)
# MyMarChemSpecCO2.update_cca("Ca", "H", "HCO3", prm.psi_Ca_H_HCO3_HMW84)
# MyMarChemSpecCO2.update_cca("Ca", "H", "CO3", prm.psi_Ca_H_CO3_HMW84)
# MyMarChemSpecCO2.update_cca("Mg", "MgOH", "HCO3", prm.psi_Mg_MgOH_HCO3_HMW84)
# MyMarChemSpecCO2.update_cca("Mg", "MgOH", "CO3", prm.psi_Mg_MgOH_CO3_HMW84)
# MyMarChemSpecCO2.update_cca("H", "Mg", "HCO3", prm.psi_H_Mg_HCO3_HMW84)
# MyMarChemSpecCO2.update_cca("H", "Mg", "CO3", prm.psi_H_Mg_CO3_HMW84)
# MyMarChemSpecCO2.update_cca("H", "MgOH", "HCO3", prm.psi_H_MgOH_HCO3_HMW84)
# MyMarChemSpecCO2.update_cca("H", "MgOH", "CO3", prm.psi_H_MgOH_CO3_HMW84)
# lambda
MyMarChemSpecCO2.update_nc("CO2", "H", prm.lambd_CO2_H_HM93)
MyMarChemSpecCO2.update_nc("CO2", "Na", prm.lambd_CO2_Na_HM93)
# MyMarChemSpecCO2.update_nc("CO2", "K", prm.lambd_CO2_K_HM93)
# MyMarChemSpecCO2.update_nc("CO2", "Ca", prm.lambd_CO2_Ca_HM93)
# MyMarChemSpecCO2.update_nc("CO2", "Mg", prm.lambd_CO2_Mg_HM93)
MyMarChemSpecCO2.update_na("CO2", "Cl", prm.lambd_CO2_Cl_HM93)
# MyMarChemSpecCO2.update_na("CO2", "SO4", prm.lambd_CO2_SO4_HM93)
# MyMarChemSpecCO2.update_na("CO2", "HSO4", prm.lambd_CO2_HSO4_HMW84)
# zeta
MyMarChemSpecCO2.update_nca("CO2", "H", "Cl", prm.zeta_CO2_H_Cl_HM93)
MyMarChemSpecCO2.update_nca("CO2", "Na", "Cl", prm.zeta_CO2_Na_Cl_HM93)
# MyMarChemSpecCO2.update_nca("CO2", "K", "Cl", prm.zeta_CO2_K_Cl_HM93)
# MyMarChemSpecCO2.update_nca("CO2", "Mg", "Cl", prm.zeta_CO2_Mg_Cl_HM93)
# MyMarChemSpecCO2.update_nca("CO2", "Ca", "Cl", prm.zeta_CO2_Ca_Cl_HM93)
# MyMarChemSpecCO2.update_nca("CO2", "H", "SO4", prm.zeta_CO2_H_SO4_HM93)
# MyMarChemSpecCO2.update_nca("CO2", "Na", "SO4", prm.zeta_CO2_Na_SO4_HM93)
# MyMarChemSpecCO2.update_nca("CO2", "K", "SO4", prm.zeta_CO2_K_SO4_HM93)
# MyMarChemSpecCO2.update_nca("CO2", "Mg", "SO4", prm.zeta_CO2_Mg_SO4_HM93)
