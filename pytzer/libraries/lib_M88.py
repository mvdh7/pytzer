# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2024  M.P. Humphreys  (GNU GPLv3)
"""Møller (1988).  System: Na-Ca-Cl-SO4.
*Geochimica et Cosmochimica Acta* 52, 821--837.
doi:10.1016/0016-7037(88)90354-7
"""

from . import Library
from .. import debyehueckel, parameters as p, unsymmetrical

library = Library(name="M88")
library.update_Aphi(debyehueckel.Aosm_M88)
library.update_ca("Ca", "Cl", p.bC_Ca_Cl_M88)
library.update_ca("Ca", "SO4", p.bC_Ca_SO4_M88)
library.update_ca("Na", "Cl", p.bC_Na_Cl_M88)
library.update_ca("Na", "SO4", p.bC_Na_SO4_M88)
library.update_cc("Ca", "Na", p.theta_Ca_Na_M88)
library.update_aa("Cl", "SO4", p.theta_Cl_SO4_M88)
library.update_cca("Ca", "Na", "Cl", p.psi_Ca_Na_Cl_M88)
library.update_cca("Ca", "Na", "SO4", p.psi_Ca_Na_SO4_M88)
library.update_caa("Ca", "Cl", "SO4", p.psi_Ca_Cl_SO4_M88)
library.update_caa("Na", "Cl", "SO4", p.psi_Na_Cl_SO4_M88)
library.update_func_J(unsymmetrical.Harvie)
