# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2020  Matthew Paul Humphreys  (GNU GPLv3)
import copy
from .. import debyehueckel, parameters as prm, unsymmetrical
from .Waters13 import Waters13

Waters13_MarChemSpec25 = copy.deepcopy(Waters13)
Waters13_MarChemSpec25.update({"name": "Waters13_MarChemSpec25"})
Waters13_MarChemSpec25.update_Aphi(debyehueckel.Aosm_MarChemSpec25)
Waters13_MarChemSpec25.assign_func_J(unsymmetrical.P75_eq47)
Waters13_MarChemSpec25.update_cc("H", "Na", prm.theta_H_Na_MarChemSpec25)
Waters13_MarChemSpec25.update_cc("H", "K", prm.theta_H_K_MarChemSpec25)
Waters13_MarChemSpec25.update_cc("Ca", "H", prm.theta_Ca_H_MarChemSpec)
Waters13_MarChemSpec25.update_cca("Mg", "MgOH", "Cl", prm.psi_Mg_MgOH_Cl_HMW84)
