# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
from .. import parameters as prm
from .. import debyehueckel, dissociation
# Start from MIAMI plus MarChemSpec equilibria and mu
from . import MarChemSpec, MIAMI
name = 'Seawater'
dh = {'Aosm': debyehueckel.Aosm_AW90}
jfunc = MIAMI.jfunc
bC = MIAMI.bC
bC['Na-Cl'] = prm.bC_Na_Cl_A92ii
bC['K-Cl'] = prm.bC_K_Cl_ZD17
theta = MIAMI.theta
psi = MIAMI.psi
lambd = MIAMI.lambd
zeta = MIAMI.zeta
mu = MarChemSpec.mu
lnk = MarChemSpec.lnk
lnk['H2CO3'] = dissociation.H2CO3_MP98
lnk['HCO3'] = dissociation.HCO3_MP98
