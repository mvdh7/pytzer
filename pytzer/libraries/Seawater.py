# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
from .. import parameters as prm
from .. import debyehueckel
# MarChemSpec project
from . import MarChemSpec
name = 'Seawater'
dh = {'Aosm': debyehueckel.Aosm_AW90}
jfunc = MarChemSpec.jfunc
bC = MarChemSpec.bC
bC['Na-Cl'] = prm.bC_Na_Cl_A92ii
bC['K-Cl'] = prm.bC_K_Cl_ZD17
theta = MarChemSpec.theta
psi = MarChemSpec.psi
lambd = MarChemSpec.lambd
zeta = MarChemSpec.zeta
mu = MarChemSpec.mu
lnk = MarChemSpec.lnk
