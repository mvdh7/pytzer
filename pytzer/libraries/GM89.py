# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
from .. import parameters as prm
from .. import debyehueckel, unsymmetrical
# Greenberg & Møller (1988). Geochim. Cosmochim. Acta 53, 2503-2518,
#  doi:10.1016/0016-7037(89)90124-5
# System: Na-K-Ca-Cl-SO4
name = 'GM89'
dh = {'Aosm': debyehueckel.Aosm_M88}
bC = {}
bC['Ca-Cl'] = prm.bC_Ca_Cl_GM89
bC['Ca-SO4'] = prm.bC_Ca_SO4_M88
bC['K-Cl'] = prm.bC_K_Cl_GM89
bC['K-SO4'] = prm.bC_K_SO4_GM89
bC['Na-Cl'] = prm.bC_Na_Cl_M88
bC['Na-SO4'] = prm.bC_Na_SO4_M88
theta = {}
theta['Ca-K'] = prm.theta_Ca_K_GM89
theta['Ca-Na'] = prm.theta_Ca_Na_M88
theta['K-Na'] = prm.theta_K_Na_GM89
theta['Cl-SO4'] = prm.theta_Cl_SO4_M88
psi = {}
psi['Ca-K-Cl'] = prm.psi_Ca_K_Cl_GM89
psi['Ca-K-SO4'] = prm.psi_Ca_K_SO4_GM89
psi['Ca-Na-Cl'] = prm.psi_Ca_Na_Cl_M88
psi['Ca-Na-SO4'] = prm.psi_Ca_Na_SO4_M88
psi['K-Na-Cl'] = prm.psi_K_Na_Cl_GM89
psi['K-Na-SO4'] = prm.psi_K_Na_SO4_GM89
psi['Ca-Cl-SO4'] = prm.psi_Ca_Cl_SO4_M88
psi['K-Cl-SO4'] = prm.psi_K_Cl_SO4_GM89
psi['Na-Cl-SO4'] = prm.psi_Na_Cl_SO4_M88
jfunc = unsymmetrical.Harvie
