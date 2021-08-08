# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2021  Matthew P. Humphreys  (GNU GPLv3)
"""Clegg et al. (1994).  System: H-HSO4-SO4.
*Journal of the Chemical Society, Faraday Transactions* 90, 1875--1894.
doi:10.1039/FT9949001875
"""
from . import ParameterLibrary
from .. import debyehueckel, dissociation as k, parameters as prm, unsymmetrical

Clegg94 = ParameterLibrary(name="Clegg94")
Clegg94.update_Aphi(debyehueckel.Aosm_CRP94)
Clegg94.update_ca("H", "HSO4", prm.bC_H_HSO4_CRP94)
Clegg94.update_ca("H", "SO4", prm.bC_H_SO4_CRP94)
Clegg94.update_aa("HSO4", "SO4", prm.theta_HSO4_SO4_CRP94)
Clegg94.update_caa("H", "HSO4", "SO4", prm.psi_H_HSO4_SO4_CRP94)
Clegg94.assign_func_J(unsymmetrical.P75_eq47)
Clegg94.update_equilibrium("HSO4", k.HSO4_CRP94)
