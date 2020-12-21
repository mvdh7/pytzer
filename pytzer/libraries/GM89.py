# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2020  Matthew Paul Humphreys  (GNU GPLv3)
from .. import parameters as prm
from .. import debyehueckel, unsymmetrical

# Greenberg & Møller (1988). Geochim. Cosmochim. Acta 53, 2503-2518,
#  doi:10.1016/0016-7037(89)90124-5
# System: Na-K-Ca-Cl-SO4
name = "GM89"
plname.update_Aphi(debyehueckel.Aosm_M88)
bC = {}
plname.update_ca("Ca", "Cl", prm.bC_Ca_Cl_GM89)
plname.update_ca("Ca", "SO4", prm.bC_Ca_SO4_M88)
plname.update_ca("K", "Cl", prm.bC_K_Cl_GM89)
plname.update_ca("K", "SO4", prm.bC_K_SO4_GM89)
plname.update_ca("Na", "Cl", prm.bC_Na_Cl_M88)
plname.update_ca("Na", "SO4", prm.bC_Na_SO4_M88)
theta = {}
plname.update_xx("Ca", "K", prm.theta_Ca_K_GM89)
plname.update_xx("Ca", "Na", prm.theta_Ca_Na_M88)
plname.update_xx("K", "Na", prm.theta_K_Na_GM89)
plname.update_xx("Cl", "SO4", prm.theta_Cl_SO4_M88)
psi = {}
plname.update_cxa("Ca", "K", "Cl", prm.psi_Ca_K_Cl_GM89)
plname.update_cxa("Ca", "K", "SO4", prm.psi_Ca_K_SO4_GM89)
plname.update_cxa("Ca", "Na", "Cl", prm.psi_Ca_Na_Cl_M88)
plname.update_cxa("Ca", "Na", "SO4", prm.psi_Ca_Na_SO4_M88)
plname.update_cxa("K", "Na", "Cl", prm.psi_K_Na_Cl_GM89)
plname.update_cxa("K", "Na", "SO4", prm.psi_K_Na_SO4_GM89)
plname.update_cxa("Ca", "Cl", "SO4", prm.psi_Ca_Cl_SO4_M88)
plname.update_cxa("K", "Cl", "SO4", prm.psi_K_Cl_SO4_GM89)
plname.update_cxa("Na", "Cl", "SO4", prm.psi_Na_Cl_SO4_M88)
plname.assign_func_J(unsymmetrical.Harvie)
