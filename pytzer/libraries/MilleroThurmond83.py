# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2023  Matthew P. Humphreys  (GNU GPLv3)
from . import ParameterLibrary
from .. import debyehueckel, dissociation as k, parameters as prm, unsymmetrical

MilleroThurmond83 = ParameterLibrary(name="MilleroThurmond83")
MilleroThurmond83.update_Aphi(debyehueckel.Aosm_M88)
MilleroThurmond83.assign_func_J(unsymmetrical.Harvie)
# Table III, part 1
MilleroThurmond83.update_ca("H", "Cl", prm.bC_H_Cl_PM73)
MilleroThurmond83.update_ca("Na", "Cl", prm.bC_Na_Cl_PM73)
MilleroThurmond83.update_ca("Mg", "Cl", prm.bC_Mg_Cl_PM73)
MilleroThurmond83.update_ca("Na", "HCO3", prm.bC_Na_HCO3_PP82)
MilleroThurmond83.update_ca("Na", "CO3", prm.bC_Na_CO3_PP82)
# Table III, part 2
MilleroThurmond83.update_cc("H", "Na", prm.theta_H_Na_PK74)
MilleroThurmond83.update_cca("H", "Na", "Cl", prm.psi_H_Na_Cl_RGB80)
MilleroThurmond83.update_cc("H", "Mg", prm.theta_H_Mg_RGB80)
MilleroThurmond83.update_cca("H", "Mg", "Cl", prm.psi_H_Na_Cl_PK74)
MilleroThurmond83.update_aa("Cl", "HCO3", prm.theta_HCO3_Cl_PP82)
MilleroThurmond83.update_caa("Na", "Cl", "HCO3", prm.psi_Na_Cl_HCO3_PP82)
MilleroThurmond83.update_aa("Cl", "CO3", prm.theta_CO3_Cl_PP82)
MilleroThurmond83.update_caa("Na", "Cl", "CO3", prm.psi_Na_CO3_Cl_TM82)
MilleroThurmond83.update_cc("Na", "Mg", prm.theta_Mg_Na_HMW84)
MilleroThurmond83.update_cca("Na", "Mg", "Cl", prm.psi_Mg_Na_Cl_HMW84)
