# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019--2020  Matthew Paul Humphreys  (GNU GPLv3)
import copy
from .. import parameters as prm
from .Waters13_MarChemSpec25 import Waters13_MarChemSpec25

MarChemSpec25 = copy.deepcopy(Waters13_MarChemSpec25)
MarChemSpec25.update({"name": "MarChemSpec25"})
# Begin with WM13_MarChemSpec25 with Aosm fixed at 25 degC
# Add parameters from GT17 Supp. Info. Table S6 (simultaneous optimisation)
# MarChemSpec25.update_ca("Na", "Cl", prm.bC_Na_Cl_GT17simopt)  # but not this one
MarChemSpec25.update_ca("trisH", "SO4", prm.bC_trisH_SO4_GT17simopt)
MarChemSpec25.update_ca("trisH", "Cl", prm.bC_trisH_Cl_GT17simopt)
MarChemSpec25.update_cc("H", "trisH", prm.theta_H_trisH_GT17simopt)
MarChemSpec25.update_cca("H", "trisH", "Cl", prm.psi_H_trisH_Cl_GT17simopt)
MarChemSpec25.update_nc("tris", "trisH", prm.lambd_tris_trisH_GT17simopt)
MarChemSpec25.update_nc("tris", "Na", prm.lambd_tris_Na_GT17simopt)
MarChemSpec25.update_nc("tris", "K", prm.lambd_tris_K_GT17simopt)
MarChemSpec25.update_nc("tris", "Mg", prm.lambd_tris_Mg_GT17simopt)
MarChemSpec25.update_nc("tris", "Ca", prm.lambd_tris_Ca_GT17simopt)
MarChemSpec25.update_nn("tris", "tris", prm.lambd_tris_tris_MarChemSpec25)
MarChemSpec25.update_nca("tris", "Na", "Cl", prm.zeta_tris_Na_Cl_MarChemSpec25)
MarChemSpec25.update_nnn("tris", prm.mu_tris_tris_tris_MarChemSpec25)
