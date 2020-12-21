# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2020  Matthew Paul Humphreys  (GNU GPLv3)
from copy import deepcopy
from .. import parameters as prm
from .. import debyehueckel  # , dissociation

# Start from MIAMI plus MarChemSpec equilibria and mu
from . import MarChemSpec, MIAMI

name = "Seawater"
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
plname.update_Aphi(debyehueckel.Aosm_AW90)
plname.update_ca("Na", "Cl", prm.bC_Na_Cl_A92ii)
plname.update_ca("K", "Cl", prm.bC_K_Cl_ZD17)
plname.update_ca("Na", "HCO3", prm.bC_Na_HCO3_HM93)
plname.update_ca("K", "HCO3", prm.bC_K_HCO3_HM93)
plname.update_ca("Mg", "HCO3", prm.bC_Mg_HCO3_HM93)
plname.update_ca("Ca", "HCO3", prm.bC_Ca_HCO3_HM93)
plname.update_ca("Na", "CO3", prm.bC_Na_CO3_HM93)
plname.update_ca("K", "CO3", prm.bC_K_CO3_HM93)
plname.update_ca("Mg", "CO3", prm.bC_Mg_CO3_HM93)
plname.update_ca("Ca", "CO3", prm.bC_Ca_CO3_HM93)
plname.update_ca("H", "Br", prm.bC_H_Br_JESS)
plname.update_ca("H", "Cl", prm.bC_H_Cl_JESS)
plname.update_ca("K", "Br", prm.bC_K_Br_JESS)
plname.update_ca("K", "Cl", prm.bC_K_Cl_JESS)
plname.update_ca("K", "OH", prm.bC_K_OH_JESS)
plname.update_ca("Na", "Br", prm.bC_Na_Br_JESS)
# Extend MIAMI/MarChemSpec:
