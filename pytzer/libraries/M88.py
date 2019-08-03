# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
from .. import parameters as prm
from .. import debyehueckel, dissociation, unsymmetrical
# Møller (1988). Geochim. Cosmochim. Acta 52, 821-837,
#  doi:10.1016/0016-7037(88)90354-7
# System: Na-Ca-Cl-SO4
name = 'M88'
dh = {'Aosm': debyehueckel.Aosm_M88}
bC = {}
bC['Ca-Cl' ] = prm.bC_Ca_Cl_M88
bC['Ca-SO4'] = prm.bC_Ca_SO4_M88
bC['Na-Cl' ] = prm.bC_Na_Cl_M88
bC['Na-SO4'] = prm.bC_Na_SO4_M88
theta = {}
theta['Ca-Na' ] = prm.theta_Ca_Na_M88
theta['Cl-SO4'] = prm.theta_Cl_SO4_M88
psi = {}
psi['Ca-Na-Cl' ] = prm.psi_Ca_Na_Cl_M88
psi['Ca-Na-SO4'] = prm.psi_Ca_Na_SO4_M88
psi['Ca-Cl-SO4'] = prm.psi_Ca_Cl_SO4_M88
psi['Na-Cl-SO4'] = prm.psi_Na_Cl_SO4_M88
jfunc = unsymmetrical.Harvie
lnk = {'H2O': dissociation.H2O_M88}
