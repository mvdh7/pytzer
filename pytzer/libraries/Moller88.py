# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
"""Møller (1988).  System: Na-Ca-Cl-SO4.
*Geochimica et Cosmochimica Acta* 52, 821--837.
doi:10.1016/0016-7037(88)90354-7
"""
from . import ParameterLibrary
from .. import debyehueckel, dissociation as k, parameters as prm, unsymmetrical

Moller88 = ParameterLibrary(name="Moller88")
Moller88.update_Aphi(debyehueckel.Aosm_M88)
Moller88.update_ca("Ca", "Cl", prm.bC_Ca_Cl_M88)
Moller88.update_ca("Ca", "SO4", prm.bC_Ca_SO4_M88)
Moller88.update_ca("Na", "Cl", prm.bC_Na_Cl_M88)
Moller88.update_ca("Na", "SO4", prm.bC_Na_SO4_M88)
Moller88.update_cc("Ca", "Na", prm.theta_Ca_Na_M88)
Moller88.update_aa("Cl", "SO4", prm.theta_Cl_SO4_M88)
Moller88.update_cca("Ca", "Na", "Cl", prm.psi_Ca_Na_Cl_M88)
Moller88.update_cca("Ca", "Na", "SO4", prm.psi_Ca_Na_SO4_M88)
Moller88.update_caa("Ca", "Cl", "SO4", prm.psi_Ca_Cl_SO4_M88)
Moller88.update_caa("Na", "Cl", "SO4", prm.psi_Na_Cl_SO4_M88)
Moller88.assign_func_J(unsymmetrical.Harvie)
Moller88.update_equilibrium("H2O", k.H2O_M88)
