# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
from copy import deepcopy
from .. import debyehueckel, dissociation
# MarChemSpec project
from . import MarChemSpec25
name = 'MarChemSpec'
dh = {'Aosm': debyehueckel.Aosm_MarChemSpec}
jfunc = deepcopy(MarChemSpec25.jfunc)
bC = deepcopy(MarChemSpec25.bC)
theta = deepcopy(MarChemSpec25.theta)
psi = deepcopy(MarChemSpec25.psi)
lambd = deepcopy(MarChemSpec25.lambd)
zeta = deepcopy(MarChemSpec25.zeta)
mu = deepcopy(MarChemSpec25.mu)
# Add equilibrium constants
lnk = {}
lnk['H2O'] = dissociation.H2O_MF
lnk['HSO4'] = dissociation.HSO4_CRP94
lnk['MgOH'] = dissociation.MgOH_CW91
lnk['trisH'] = dissociation.trisH_BH64
