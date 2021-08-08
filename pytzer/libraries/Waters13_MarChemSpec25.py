# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
from . import ParameterLibrary
from .. import debyehueckel, dissociation as k, parameters as prm, unsymmetrical

Waters13_MarChemSpec25 = ParameterLibrary(name="Waters13_MarChemSpec25")
Waters13_MarChemSpec25.update_Aphi(debyehueckel.Aosm_MarChemSpec25)
Waters13_MarChemSpec25.assign_func_J(unsymmetrical.P75_eq47)
# Table A1: Na salts
Waters13_MarChemSpec25.update_ca("Na", "Cl", prm.bC_Na_Cl_M88)
Waters13_MarChemSpec25.update_ca("Na", "SO4", prm.bC_Na_SO4_HM86)
Waters13_MarChemSpec25.update_ca("Na", "HSO4", prm.bC_Na_HSO4_HPR93)
Waters13_MarChemSpec25.update_ca("Na", "OH", prm.bC_Na_OH_PP87i)
# Table A2: Mg salts
Waters13_MarChemSpec25.update_ca("Mg", "Cl", prm.bC_Mg_Cl_dLP83)
Waters13_MarChemSpec25.update_ca("Mg", "SO4", prm.bC_Mg_SO4_PP86ii)
Waters13_MarChemSpec25.update_ca("Mg", "HSO4", prm.bC_Mg_HSO4_RC99)
# Table A3: Ca salts
Waters13_MarChemSpec25.update_ca("Ca", "Cl", prm.bC_Ca_Cl_GM89)
Waters13_MarChemSpec25.update_ca("Ca", "SO4", prm.bC_Ca_SO4_WM13)
Waters13_MarChemSpec25.update_ca("Ca", "HSO4", prm.bC_Ca_HSO4_WM13)
Waters13_MarChemSpec25.update_ca("Ca", "OH", prm.bC_Ca_OH_HMW84)
# Table A4: K salts
Waters13_MarChemSpec25.update_ca("K", "Cl", prm.bC_K_Cl_GM89)
Waters13_MarChemSpec25.update_ca("K", "SO4", prm.bC_K_SO4_HM86)
Waters13_MarChemSpec25.update_ca("K", "HSO4", prm.bC_K_HSO4_WM13)
Waters13_MarChemSpec25.update_ca("K", "OH", prm.bC_K_OH_HMW84)
# Table A5: H+ interactions
Waters13_MarChemSpec25.update_ca("H", "Cl", prm.bC_H_Cl_CMR93)
Waters13_MarChemSpec25.update_ca("H", "SO4", prm.bC_H_SO4_CRP94)
Waters13_MarChemSpec25.update_ca("H", "HSO4", prm.bC_H_HSO4_CRP94)
# Table A6: MgOH+ interactions
Waters13_MarChemSpec25.update_ca("MgOH", "Cl", prm.bC_MgOH_Cl_HMW84)
# Table A7: cation-cation interactions
Waters13_MarChemSpec25.update_cc("H", "Na", prm.theta_H_Na_MarChemSpec25)
Waters13_MarChemSpec25.update_cc("H", "Mg", prm.theta_H_Mg_RGB80)
Waters13_MarChemSpec25.update_cc("Ca", "H", prm.theta_Ca_H_MarChemSpec)
Waters13_MarChemSpec25.update_cc("H", "K", prm.theta_H_K_MarChemSpec25)
Waters13_MarChemSpec25.update_cc("Mg", "Na", prm.theta_Mg_Na_HMW84)
Waters13_MarChemSpec25.update_cc("Ca", "Na", prm.theta_Ca_Na_HMW84)
Waters13_MarChemSpec25.update_cc("K", "Na", prm.theta_K_Na_HMW84)
Waters13_MarChemSpec25.update_cc("Ca", "Mg", prm.theta_Ca_Mg_HMW84)
Waters13_MarChemSpec25.update_cc("K", "Mg", prm.theta_K_Mg_HMW84)
Waters13_MarChemSpec25.update_cc("Ca", "K", prm.theta_Ca_K_HMW84)
# Table A7: anion-anion interactions
Waters13_MarChemSpec25.update_aa("Cl", "SO4", prm.theta_Cl_SO4_HMW84)
Waters13_MarChemSpec25.update_aa("Cl", "HSO4", prm.theta_Cl_HSO4_HMW84)
Waters13_MarChemSpec25.update_aa("Cl", "OH", prm.theta_Cl_OH_HMW84)
Waters13_MarChemSpec25.update_aa("HSO4", "SO4", prm.theta_HSO4_SO4_WM13)
Waters13_MarChemSpec25.update_aa("OH", "SO4", prm.theta_OH_SO4_HMW84)
# Table A8: c-a-a' triplets
Waters13_MarChemSpec25.update_caa(
    "H", "Cl", "SO4", prm.psi_H_Cl_SO4_WM13
)  # agrees with HMW84
Waters13_MarChemSpec25.update_caa("Na", "Cl", "SO4", prm.psi_Na_Cl_SO4_HMW84)
Waters13_MarChemSpec25.update_caa("Mg", "Cl", "SO4", prm.psi_Mg_Cl_SO4_HMW84)
Waters13_MarChemSpec25.update_caa("Ca", "Cl", "SO4", prm.psi_Ca_Cl_SO4_HMW84)
Waters13_MarChemSpec25.update_caa("K", "Cl", "SO4", prm.psi_K_Cl_SO4_HMW84)
Waters13_MarChemSpec25.update_caa("H", "Cl", "HSO4", prm.psi_H_Cl_HSO4_HMW84)
Waters13_MarChemSpec25.update_caa("Na", "Cl", "HSO4", prm.psi_Na_Cl_HSO4_HMW84)
Waters13_MarChemSpec25.update_caa("Mg", "Cl", "HSO4", prm.psi_Mg_Cl_HSO4_HMW84)
Waters13_MarChemSpec25.update_caa("Ca", "Cl", "HSO4", prm.psi_Ca_Cl_HSO4_HMW84)
Waters13_MarChemSpec25.update_caa("K", "Cl", "HSO4", prm.psi_K_Cl_HSO4_HMW84)
Waters13_MarChemSpec25.update_caa(
    "H", "Cl", "OH", prm.psi_H_Cl_OH_WM13
)  # agrees with HMW84
Waters13_MarChemSpec25.update_caa("Na", "Cl", "OH", prm.psi_Na_Cl_OH_HMW84)
Waters13_MarChemSpec25.update_caa(
    "Mg", "Cl", "OH", prm.psi_Mg_Cl_OH_WM13
)  # agrees with HMW84
Waters13_MarChemSpec25.update_caa("Ca", "Cl", "OH", prm.psi_Ca_Cl_OH_HMW84)
Waters13_MarChemSpec25.update_caa("K", "Cl", "OH", prm.psi_K_Cl_OH_HMW84)
Waters13_MarChemSpec25.update_caa("H", "HSO4", "SO4", prm.psi_H_HSO4_SO4_HMW84)
Waters13_MarChemSpec25.update_caa("Na", "HSO4", "SO4", prm.psi_Na_HSO4_SO4_HMW84)
Waters13_MarChemSpec25.update_caa("Mg", "HSO4", "SO4", prm.psi_Mg_HSO4_SO4_RC99)
Waters13_MarChemSpec25.update_caa(
    "Ca", "HSO4", "SO4", prm.psi_Ca_HSO4_SO4_WM13
)  # agrees with HMW84
Waters13_MarChemSpec25.update_caa("K", "HSO4", "SO4", prm.psi_K_HSO4_SO4_HMW84)
Waters13_MarChemSpec25.update_caa(
    "H", "OH", "SO4", prm.psi_H_OH_SO4_WM13
)  # agrees with HMW84
Waters13_MarChemSpec25.update_caa("Na", "OH", "SO4", prm.psi_Na_OH_SO4_HMW84)
Waters13_MarChemSpec25.update_caa(
    "Mg", "OH", "SO4", prm.psi_Mg_OH_SO4_WM13
)  # agrees with HMW84
Waters13_MarChemSpec25.update_caa(
    "Ca", "OH", "SO4", prm.psi_Ca_OH_SO4_WM13
)  # agrees with HMW84
Waters13_MarChemSpec25.update_caa("K", "OH", "SO4", prm.psi_K_OH_SO4_HMW84)
# Table A9: c-c'-a triplets
Waters13_MarChemSpec25.update_cca("H", "Na", "Cl", prm.psi_H_Na_Cl_HMW84)
Waters13_MarChemSpec25.update_cca(
    "H", "Na", "SO4", prm.psi_H_Na_SO4_WM13
)  # agrees with HMW84
Waters13_MarChemSpec25.update_cca("H", "Na", "HSO4", prm.psi_H_Na_HSO4_HMW84)
Waters13_MarChemSpec25.update_cca("H", "Mg", "Cl", prm.psi_H_Mg_Cl_HMW84)
Waters13_MarChemSpec25.update_cca("H", "Mg", "SO4", prm.psi_H_Mg_SO4_RC99)
Waters13_MarChemSpec25.update_cca("H", "Mg", "HSO4", prm.psi_H_Mg_HSO4_RC99)
Waters13_MarChemSpec25.update_cca("Ca", "H", "Cl", prm.psi_Ca_H_Cl_HMW84)
Waters13_MarChemSpec25.update_cca(
    "Ca", "H", "SO4", prm.psi_Ca_H_SO4_WM13
)  # agrees with HMW84
Waters13_MarChemSpec25.update_cca(
    "Ca", "H", "HSO4", prm.psi_Ca_H_HSO4_WM13
)  # agrees with HMW84
Waters13_MarChemSpec25.update_cca("H", "K", "Cl", prm.psi_H_K_Cl_HMW84)
Waters13_MarChemSpec25.update_cca("H", "K", "SO4", prm.psi_H_K_SO4_HMW84)
Waters13_MarChemSpec25.update_cca("H", "K", "HSO4", prm.psi_H_K_HSO4_HMW84)
Waters13_MarChemSpec25.update_cca("Mg", "Na", "Cl", prm.psi_Mg_Na_Cl_HMW84)
Waters13_MarChemSpec25.update_cca("Mg", "Na", "SO4", prm.psi_Mg_Na_SO4_HMW84)
Waters13_MarChemSpec25.update_cca(
    "Mg", "Na", "HSO4", prm.psi_Mg_Na_HSO4_WM13
)  # agrees with HMW84
Waters13_MarChemSpec25.update_cca("Ca", "Na", "Cl", prm.psi_Ca_Na_Cl_HMW84)
Waters13_MarChemSpec25.update_cca("Ca", "Na", "SO4", prm.psi_Ca_Na_SO4_HMW84)
Waters13_MarChemSpec25.update_cca(
    "Ca", "Na", "HSO4", prm.psi_Ca_Na_HSO4_WM13
)  # agrees with HMW84
Waters13_MarChemSpec25.update_cca("K", "Na", "Cl", prm.psi_K_Na_Cl_HMW84)
Waters13_MarChemSpec25.update_cca("K", "Na", "SO4", prm.psi_K_Na_SO4_HMW84)
Waters13_MarChemSpec25.update_cca(
    "K", "Na", "HSO4", prm.psi_K_Na_HSO4_WM13
)  # agrees with HMW84
Waters13_MarChemSpec25.update_cca("Ca", "Mg", "Cl", prm.psi_Ca_Mg_Cl_HMW84)
Waters13_MarChemSpec25.update_cca("Ca", "Mg", "SO4", prm.psi_Ca_Mg_SO4_HMW84)
Waters13_MarChemSpec25.update_cca(
    "Ca", "Mg", "HSO4", prm.psi_Ca_Mg_HSO4_WM13
)  # agrees with HMW84
Waters13_MarChemSpec25.update_cca("K", "Mg", "Cl", prm.psi_K_Mg_Cl_HMW84)
Waters13_MarChemSpec25.update_cca("K", "Mg", "SO4", prm.psi_K_Mg_SO4_HMW84)
Waters13_MarChemSpec25.update_cca(
    "K", "Mg", "HSO4", prm.psi_K_Mg_HSO4_WM13
)  # agrees with HMW84
Waters13_MarChemSpec25.update_cca("Ca", "K", "Cl", prm.psi_Ca_K_Cl_HMW84)
Waters13_MarChemSpec25.update_cca(
    "Ca", "K", "SO4", prm.psi_Ca_K_SO4_WM13
)  # agrees with HMW84
Waters13_MarChemSpec25.update_cca(
    "Ca", "K", "HSO4", prm.psi_Ca_K_HSO4_WM13
)  # agrees with HMW84
Waters13_MarChemSpec25.update_cca("Mg", "MgOH", "Cl", prm.psi_Mg_MgOH_Cl_HMW84)
