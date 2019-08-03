# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
from .. import parameters as prm
# Waters and Millero (2013). Mar. Chem. 149, 8-22,
#  doi:10.1016/j.marchem.2012.11.003
from . import WM13_MarChemSpec25
name = 'MarChemSpec25'
dh = WM13_MarChemSpec25.dh
jfunc = WM13_MarChemSpec25.jfunc
# Begin with WM13_MarChemSpec25 with Aosm fixed at 25 degC
# Add parameters from GT17 Supp. Info. Table S6 (simultaneous optimisation)
bC = WM13_MarChemSpec25.bC
#bC['Na-Cl'] = prm.bC_Na_Cl_GT17simopt
bC['trisH-SO4'] = prm.bC_trisH_SO4_GT17simopt
bC['trisH-Cl' ] = prm.bC_trisH_Cl_GT17simopt
theta = WM13_MarChemSpec25.theta
theta['H-trisH'] = prm.theta_H_trisH_GT17simopt
psi = WM13_MarChemSpec25.psi
psi['H-trisH-Cl'] = prm.psi_H_trisH_Cl_GT17simopt
lambd = {}
lambd['tris-trisH'] = prm.lambd_tris_trisH_GT17simopt
lambd['tris-Na'] = prm.lambd_tris_Na_GT17simopt
lambd['tris-K'] = prm.lambd_tris_K_GT17simopt
lambd['tris-Mg'] = prm.lambd_tris_Mg_GT17simopt
lambd['tris-Ca'] = prm.lambd_tris_Ca_GT17simopt
lambd['tris-tris'] = prm.lambd_tris_tris_MarChemSpec25
zeta = {'tris-Na-Cl': prm.zeta_tris_Na_Cl_MarChemSpec25}
mu = {'tris-tris-tris': prm.mu_tris_tris_tris_MarChemSpec25}
