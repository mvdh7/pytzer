# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
from copy import deepcopy
from .. import parameters as prm
from .. import debyehueckel#, dissociation
# Start from MIAMI plus MarChemSpec equilibria and mu
from . import MarChemSpec, MIAMI
name = 'Seawater'
# Copy MIAMI/MarChemSpec:
jfunc = deepcopy(MIAMI.jfunc)
bC = deepcopy(MIAMI.bC)
theta = deepcopy(MIAMI.theta)
psi = deepcopy(MIAMI.psi)
lambd = deepcopy(MIAMI.lambd)
zeta = deepcopy(MIAMI.zeta)
mu = deepcopy(MarChemSpec.mu)
lnk = deepcopy(MIAMI.lnk)
# Overwrite MIAMI/MarChemSpec:
dh = {'Aosm': debyehueckel.Aosm_AW90}
bC['Na-Cl'] = prm.bC_Na_Cl_A92ii
bC['K-Cl'] = prm.bC_K_Cl_ZD17
bC['Na-HCO3'] = prm.bC_Na_HCO3_HM93
bC['K-HCO3'] = prm.bC_K_HCO3_HM93
bC['Mg-HCO3'] = prm.bC_Mg_HCO3_HM93
bC['Ca-HCO3'] = prm.bC_Ca_HCO3_HM93
bC['Na-CO3'] = prm.bC_Na_CO3_HM93
bC['K-CO3'] = prm.bC_K_CO3_HM93
bC['Mg-CO3'] = prm.bC_Mg_CO3_HM93
bC['Ca-CO3'] = prm.bC_Ca_CO3_HM93
bC['H-Br'] = prm.bC_H_Br_JESS
bC['H-Cl'] = prm.bC_H_Cl_JESS
bC['K-Br'] = prm.bC_K_Br_JESS
bC['K-Cl'] = prm.bC_K_Cl_JESS
bC['K-OH'] = prm.bC_K_OH_JESS
bC['Na-Br'] = prm.bC_Na_Br_JESS
# Extend MIAMI/MarChemSpec:
