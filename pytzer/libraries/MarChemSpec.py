# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
from .. import debyehueckel, dissociation
# MarChemSpec project
from . import MarChemSpec25
name = 'MarChemSpec'
dh = {'Aosm': debyehueckel.Aosm_MarChemSpec}
jfunc = MarChemSpec25.jfunc
bC = MarChemSpec25.bC
theta = MarChemSpec25.theta
psi = MarChemSpec25.psi
lambd = MarChemSpec25.lambd
zeta = MarChemSpec25.zeta
mu = MarChemSpec25.mu
# Add equilibrium constants
lnk = {}
lnk['H2O'] = dissociation.H2O_MF
lnk['HSO4'] = dissociation.HSO4_CRP94
lnk['Mg'] = dissociation.Mg_CW91
lnk['trisH'] = dissociation.trisH_BH64
