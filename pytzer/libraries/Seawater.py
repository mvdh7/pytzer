# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2020  Matthew Paul Humphreys  (GNU GPLv3)
import copy
from .. import debyehueckel, parameters as prm
from .MarChemSpec import MarChemSpec
from .Millero98 import Millero98

# Start from MIAMI plus MarChemSpec equilibria and mu
Seawater = copy.deepcopy(Millero98)
Seawater.update({"name": "Seawater"})
Seawater.update({"nnn": copy.deepcopy(MarChemSpec["nnn"])})
# Overwrite MIAMI/MarChemSpec:
Seawater.update_Aphi(debyehueckel.Aosm_AW90)
Seawater.update_ca("Na", "Cl", prm.bC_Na_Cl_A92ii)
Seawater.update_ca("K", "Cl", prm.bC_K_Cl_ZD17)
Seawater.update_ca("Na", "HCO3", prm.bC_Na_HCO3_HM93)
Seawater.update_ca("K", "HCO3", prm.bC_K_HCO3_HM93)
Seawater.update_ca("Mg", "HCO3", prm.bC_Mg_HCO3_HM93)
Seawater.update_ca("Ca", "HCO3", prm.bC_Ca_HCO3_HM93)
Seawater.update_ca("Na", "CO3", prm.bC_Na_CO3_HM93)
Seawater.update_ca("K", "CO3", prm.bC_K_CO3_HM93)
Seawater.update_ca("Mg", "CO3", prm.bC_Mg_CO3_HM93)
Seawater.update_ca("Ca", "CO3", prm.bC_Ca_CO3_HM93)
Seawater.update_ca("H", "Br", prm.bC_H_Br_JESS)
Seawater.update_ca("H", "Cl", prm.bC_H_Cl_JESS)
Seawater.update_ca("K", "Br", prm.bC_K_Br_JESS)
Seawater.update_ca("K", "Cl", prm.bC_K_Cl_JESS)
Seawater.update_ca("K", "OH", prm.bC_K_OH_JESS)
Seawater.update_ca("Na", "Br", prm.bC_Na_Br_JESS)
Seawater.update_ca("H", "SO4", prm.bC_H_SO4_CRP94)
Seawater.update_ca("H", "HSO4", prm.bC_H_HSO4_CRP94)
# Extend MIAMI/MarChemSpec:
# TBC...
