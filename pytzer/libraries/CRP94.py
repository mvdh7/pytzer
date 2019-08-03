# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
from .. import parameters as prm
from .. import debyehueckel, dissociation, unsymmetrical
# Clegg et al. (1994). J. Chem. Soc., Faraday Trans. 90, 1875-1894,
#  doi:10.1039/FT9949001875
# System: H-HSO4-SO4
name = 'CRP94'
dh = {'Aosm': debyehueckel.Aosm_CRP94}
bC = {}
bC['H-HSO4'] = prm.bC_H_HSO4_CRP94
bC['H-SO4' ] = prm.bC_H_SO4_CRP94
theta = {'HSO4-SO4': prm.theta_HSO4_SO4_CRP94}
psi = {'H-HSO4-SO4': prm.psi_H_HSO4_SO4_CRP94}
jfunc = unsymmetrical.P75_eq47
lnk = {'HSO4': dissociation.HSO4_CRP94}
