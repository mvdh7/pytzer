# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
"""Greenberg & Møller (1988). System:  Na-K-Ca-Cl-SO4.
*Geochimica et Cosmochimica Acta* 53, 2503--2518.
doi:10.1016/0016-7037(89)90124-5
"""
from . import ParameterLibrary
from .. import debyehueckel, dissociation as k, parameters as prm, unsymmetrical

Greenberg89 = ParameterLibrary(name="Greenberg89")
Greenberg89.update_Aphi(debyehueckel.Aosm_M88)
Greenberg89.update_ca("Ca", "Cl", prm.bC_Ca_Cl_GM89)
Greenberg89.update_ca("Ca", "SO4", prm.bC_Ca_SO4_M88)
Greenberg89.update_ca("K", "Cl", prm.bC_K_Cl_GM89)
Greenberg89.update_ca("K", "SO4", prm.bC_K_SO4_GM89)
Greenberg89.update_ca("Na", "Cl", prm.bC_Na_Cl_M88)
Greenberg89.update_ca("Na", "SO4", prm.bC_Na_SO4_M88)
Greenberg89.update_cc("Ca", "K", prm.theta_Ca_K_GM89)
Greenberg89.update_cc("Ca", "Na", prm.theta_Ca_Na_M88)
Greenberg89.update_cc("K", "Na", prm.theta_K_Na_GM89)
Greenberg89.update_aa("Cl", "SO4", prm.theta_Cl_SO4_M88)
Greenberg89.update_cca("Ca", "K", "Cl", prm.psi_Ca_K_Cl_GM89)
Greenberg89.update_cca("Ca", "K", "SO4", prm.psi_Ca_K_SO4_GM89)
Greenberg89.update_cca("Ca", "Na", "Cl", prm.psi_Ca_Na_Cl_M88)
Greenberg89.update_cca("Ca", "Na", "SO4", prm.psi_Ca_Na_SO4_M88)
Greenberg89.update_cca("K", "Na", "Cl", prm.psi_K_Na_Cl_GM89)
Greenberg89.update_cca("K", "Na", "SO4", prm.psi_K_Na_SO4_GM89)
Greenberg89.update_caa("Ca", "Cl", "SO4", prm.psi_Ca_Cl_SO4_M88)
Greenberg89.update_caa("K", "Cl", "SO4", prm.psi_K_Cl_SO4_GM89)
Greenberg89.update_caa("Na", "Cl", "SO4", prm.psi_Na_Cl_SO4_M88)
Greenberg89.assign_func_J(unsymmetrical.Harvie)
